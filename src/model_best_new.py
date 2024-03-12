import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *




class CKAN(nn.Module):
    def __init__(self, args, n_entity, n_relation, n_user, n_item):
        super(CKAN, self).__init__()
        
        # 根据参数设置对象的属性
        self.args = args
        self._parse_args(args, n_entity, n_relation, n_user, n_item)

        # 初始化 entity 和 relation 的 embedding
        self.user_emb = nn.Embedding(self.n_user, self.dim)
        self.item_emb = nn.Embedding(self.n_item, self.dim)
        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        
        # attention 层
        self.attention = nn.Sequential(
                nn.Linear(self.dim*2, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, 1, bias=False),
                nn.Sigmoid(),
                )
        
        # gate 层
        self._knowledge_gate = nn.Sequential(
                nn.Linear(self.dim*2, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim, bias=False),
                nn.Sigmoid(),
                )


        # 初始化权重
        self._init_weight()

        # 交互层
        self.user_cc_unit = CrossCompressUnit(self.dim)
        self.item_cc_unit = CrossCompressUnit(self.dim)

        # 知识图谱模块中使用到的一些全连接层
        tail_layers = 1
        self.user_tail_mlp = nn.Sequential()
        for i_cnt in range(tail_layers):
            self.user_tail_mlp.add_module('user_tail_mlp{}'.format(i_cnt),
                                     Dense(self.dim, self.dim))

        self.item_tail_mlp = nn.Sequential()
        for i_cnt in range(tail_layers):
            self.item_tail_mlp.add_module('item_tail_mlp{}'.format(i_cnt),
                                     Dense(self.dim, self.dim))

        kgb_layers = 1
        self.user_kge_mlp = nn.Sequential()
        for i_cnt in range(kgb_layers):
            self.user_kge_mlp.add_module('user_kge_mlp{}'.format(i_cnt),
                                    Dense(self.dim * 2, self.dim))

        self.item_kge_mlp = nn.Sequential()
        for i_cnt in range(kgb_layers):
            self.item_kge_mlp.add_module('item_kge_mlp{}'.format(i_cnt),
                                    Dense(self.dim * 2, self.dim))

        # fusion_layer
        fusion_layers = 1
        self.user_fusion_mlp = nn.Sequential()
        for i_cnt in range(fusion_layers):
            self.user_fusion_mlp.add_module('user_fusion_mlp{}'.format(i_cnt),
                                     Dense(self.dim, self.dim))

        self.Contrastive = Contrast(args.temperature)



    def get_miRNA_full_emb(self, miRNAs, miRNA_triple_set):

        """ miRNA 塔 """
        miRNA_embeddings = []

        # [batch_size, triple_set_size, dim]
        miRNA_emb_0 = self.miRNA_emb(miRNA_triple_set[0][0]) if miRNA_triple_set[0][
                                                                    0].type == 'miRNA' else self.disease_emb(
            miRNA_triple_set[0][0])
        # [batch_size, dim]
        miRNA_emb_origin = miRNA_emb_0.mean(dim=1)
        miRNA_embeddings.append(miRNA_emb_origin)

        miRNA_knowledge_emb = None

        for i in range(self.n_layer):  # n_layer * [batch_size, dim]
            # [batch_size, triple_set_size, dim]
            h_emb = self.miRNA_emb(miRNA_triple_set[0][i]) if miRNA_triple_set[0][i].type == 'miRNA' else self.disease_emb(miRNA_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.m_d_relation_emb(miRNA_triple_set[1][i]) if miRNA_triple_set[1][i].type == 'm-d' else \
                self.m_m_relation_emb(miRNA_triple_set[1][i]) if miRNA_triple_set[1][i].type == 'm-m' else None
            if r_emb is None:
                continue

            # [batch_size, triple_set_size, dim]
            t_emb = self.miRNA_emb(miRNA_triple_set[2][i]) if miRNA_triple_set[2][i].type == 'miRNA' else self.disease_emb(miRNA_triple_set[2][i])
            # [batch_size, dim]
            miRNA_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            if i == 0:
                miRNA_knowledge_emb = h_emb.mean(dim=1)
            miRNA_embeddings.append(miRNA_emb_i)

        miRNA_cross_emb = miRNA_emb_origin
        if self.args.use_raw_id_emb:
            miRNA_id_emb = self.miRNA_emb(miRNAs)
            miRNA_cross_emb = miRNA_id_emb

        _, head_emb = self.miRNA_cc_unit([miRNA_cross_emb, miRNA_knowledge_emb])
        miRNA_embeddings.append(head_emb)

        miRNA_embeddings_concat = torch.cat(miRNA_embeddings, axis=-1)

        return miRNA_embeddings_concat


    def get_disease_full_emb(self, diseases, disease_triple_set):

        """ disease 塔 """
        disease_embeddings = []

        # [batch_size, triple_set_size, dim]
        disease_emb_0 = self.disease_emb(disease_triple_set[0][0]) if disease_triple_set[0][0].type == 'disease' else self.miRNA_emb(
            disease_triple_set[0][0])
        # [batch_size, dim]
        disease_emb_origin = disease_emb_0.mean(dim=1)
        disease_embeddings.append(disease_emb_origin)

        disease_knowledge_emb = None

        for i in range(self.n_layer):  # n_layer * [batch_size, dim]
            # [batch_size, triple_set_size, dim]
            h_emb = self.disease_emb(disease_triple_set[0][i]) if disease_triple_set[0][
                                                                      i].type == 'disease' else self.miRNA_emb(
                disease_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.d_m_relation_emb(disease_triple_set[1][i]) if disease_triple_set[1][i].type == 'd-m' else \
                self.d_d_relation_emb(disease_triple_set[1][i]) if disease_triple_set[1][i].type == 'd-d' else None
            if r_emb is None:
                continue
            # [batch_size, triple_set_size, dim]
            t_emb = self.disease_emb(disease_triple_set[2][i]) if disease_triple_set[2][
                                                                      i].type == 'disease' else self.miRNA_emb(
                disease_triple_set[2][i])
            # [batch_size, dim]
            disease_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            if i == 0:
                disease_knowledge_emb = h_emb.mean(dim=1)
            disease_embeddings.append(disease_emb_i)

        disease_cross_emb = disease_emb_origin
        if self.args.use_raw_id_emb:
            disease_id_emb = self.disease_emb(diseases)
            disease_cross_emb = disease_id_emb

        _, head_emb = self.disease_cc_unit([disease_cross_emb, disease_knowledge_emb])
        disease_embeddings.append(head_emb)

        disease_embeddings_concat = torch.cat(disease_embeddings, axis=-1)

        return disease_embeddings_concat

                
    def forward(
        self,
        users: torch.LongTensor,
        items: torch.LongTensor,
        miRNA_triple_set: list,
        disease_triple_set: list,
    ):       
        
        """
            items: target-item
            user_triple_set: List 类型, [h,r,t], 其中 h r t 分别都是 List 类型, 每个元素都是 LongTensor.
        """

        # 全部 user 特征拼接
        user_embeddings_concat = self.get_user_full_emb(users, miRNA_triple_set)

        # 全部 item 特征拼接
        item_embeddings_concat = self.get_item_full_emb(items, disease_triple_set)


        # 对比学习 loss
        contrastive_loss = 0
        if self.args.use_contrastive_loss:
            contrastive_loss = self.Contrastive.semi_loss(user_embeddings_concat, item_embeddings_concat)
            contrastive_loss = torch.mean(contrastive_loss)


        """ miRNA 知识图谱表征模块 """
        miRNA_cos_loss = 0
        if self.args.use_miRNA_kg_loss:
            h_emb_list = []
            r_emb_list = []
            t_emb_list = []

            for i in range(self.n_layer):  # n_layer * [batch_size, dim]
                # [batch_size, triple_set_size, dim]
                h_emb = self.miRNA_emb(miRNA_triple_set[0][i]) if miRNA_triple_set[0][i].type == 'miRNA' else self.disease_emb(
                    miRNA_triple_set[0][i])

                # [batch_size, triple_set_size, dim]
                r_emb = self.m_d_relation_emb(miRNA_triple_set[1][i]) if miRNA_triple_set[1][
                                                                             i].type == 'm-d' else self.m_m_relation_emb(
                    miRNA_triple_set[1][i]) if miRNA_triple_set[1][i].type == 'm-m' else None

                # [batch_size, triple_set_size, dim]
                t_emb = self.miRNA_emb(miRNA_triple_set[2][i]) if miRNA_triple_set[2][i].type == 'miRNA' else self.disease_emb(miRNA_triple_set[2][i])

                h_emb = torch.unsqueeze(h_emb, 1)
                r_emb = torch.unsqueeze(r_emb, 1)
                t_emb = torch.unsqueeze(t_emb, 1)

                h_emb_list.append(h_emb)
                r_emb_list.append(r_emb)
                t_emb_list.append(t_emb)

            h_emb = torch.cat(h_emb_list, axis=1)
            r_emb = torch.cat(r_emb_list, axis=1)
            t_emb = torch.cat(t_emb_list, axis=1)

            t_emb = self.miRNA_tail_mlp(t_emb)

            # [batch_size, dim * 2]
            head_relation_concat = torch.cat([h_emb, r_emb], -1)
            tail_pred = self.miRNA_kge_mlp(head_relation_concat)

            # Normalization
            tail_pred = F.normalize(tail_pred, p=2, dim=-1)
            t_emb = F.normalize(t_emb, p=2, dim=-1)
            miRNA_cos_loss = 1 - F.cosine_similarity(tail_pred, t_emb, dim=-1)
            miRNA_cos_loss = torch.mean(miRNA_cos_loss)
            miRNA_cos_loss = 0.05 * miRNA_cos_loss


        """ disease 知识图谱表征模块 """
        disease_cos_loss = 0
        if self.args.use_disease_kg_loss:
            h_emb_list = []
            r_emb_list = []
            t_emb_list = []

            for i in range(self.n_layer):  # n_layer * [batch_size, dim]
                # [batch_size, triple_set_size, dim]
                h_emb = self.disease_emb(disease_triple_set[0][i]) if disease_triple_set[0][
                                                                          i].type == 'disease' else self.miRNA_emb(
                    disease_triple_set[0][i])

                # [batch_size, triple_set_size, dim]
                r_emb = self.m_d_relation_emb(disease_triple_set[1][i]) if disease_triple_set[1][
                                                                               i].type == 'm-d' else self.d_d_relation_emb(
                    disease_triple_set[1][i]) if disease_triple_set[1][i].type == 'd-d' else None

                # [batch_size, triple_set_size, dim]
                t_emb = self.disease_emb(disease_triple_set[2][i]) if disease_triple_set[2][
                                                                          i].type == 'disease' else self.miRNA_emb(
                    disease_triple_set[2][i])

                h_emb = torch.unsqueeze(h_emb, 1)
                r_emb = torch.unsqueeze(r_emb, 1)
                t_emb = torch.unsqueeze(t_emb, 1)

                h_emb_list.append(h_emb)
                r_emb_list.append(r_emb)
                t_emb_list.append(t_emb)

            h_emb = torch.cat(h_emb_list, axis=1)
            r_emb = torch.cat(r_emb_list, axis=1)
            t_emb = torch.cat(t_emb_list, axis=1)

            t_emb = self.disease_tail_mlp(t_emb)

            # [batch_size, dim * 2]
            head_relation_concat = torch.cat([h_emb, r_emb], -1)
            tail_pred = self.disease_kge_mlp(head_relation_concat)

            tail_pred = F.normalize(tail_pred, p=2, dim=-1)
            t_emb = F.normalize(t_emb, p=2, dim=-1)
            disease_cos_loss = 1 - F.cosine_similarity(tail_pred, t_emb, dim=-1)
            disease_cos_loss = torch.mean(disease_cos_loss)

            # 双塔预测打分
            scores = self.predict([user_embeddings_concat], [item_embeddings_concat])

            return scores, contrastive_loss, miRNA_cos_loss, disease_cos_loss
    

    def predict(self, user_embeddings, item_embeddings):
        e_u = user_embeddings[0]
        e_v = item_embeddings[0]
    
        if self.agg == 'concat':
            if len(user_embeddings) != len(item_embeddings):
                raise Exception("Concat aggregator needs same length for user and item embedding")
            for i in range(1, len(user_embeddings)):
                e_u = torch.cat((user_embeddings[i], e_u),dim=-1)
            for i in range(1, len(item_embeddings)):
                e_v = torch.cat((item_embeddings[i], e_v),dim=-1)
        elif self.agg == 'sum':
            for i in range(1, len(user_embeddings)):
                e_u += user_embeddings[i]
            for i in range(1, len(item_embeddings)):
                e_v += item_embeddings[i]
        elif self.agg == 'pool':
            for i in range(1, len(user_embeddings)):
                e_u = torch.max(e_u, user_embeddings[i])
            for i in range(1, len(item_embeddings)):
                e_v = torch.max(e_v, item_embeddings[i])
        else:
            raise Exception("Unknown aggregator: " + self.agg)

        scores = (e_v * e_u).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores
    
    
    def _parse_args(self, args, n_entity, n_relation, n_user, n_item):
        """ 解析输入的参数, 将其设置为对象的属性 """

        self.n_user = n_user
        self.n_item = n_item

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_layer = args.n_layer
        self.agg = args.agg
        
        
    def _init_weight(self):
        """ 采用 xavier_uniform 初始化所有权重 """
        # init embedding
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        # init attention
        for layer in self.attention:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    
    def _knowledge_attention(self, h_emb, r_emb, t_emb):
        # [batch_size, triple_set_size]
        att_weights = self.attention(torch.cat((h_emb,r_emb),dim=-1)).squeeze(-1)
        # [batch_size, triple_set_size]
        att_weights_norm = F.softmax(att_weights,dim=-1)
        # [batch_size, triple_set_size, dim]
        emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)

        knowledge_gate_output = 2 * self._knowledge_gate(torch.cat((h_emb,r_emb),dim=-1))
        emb_i = knowledge_gate_output * emb_i

        # [batch_size, dim]
        emb_i = emb_i.sum(dim=1)
        return emb_i