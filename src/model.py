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


    def get_user_full_emb(self, users, user_triple_set):

        """ user 塔 """
        user_embeddings = []
        
        # [batch_size, triple_set_size, dim]
        user_emb_0 = self.entity_emb(user_triple_set[0][0])
        # [batch_size, dim]
        user_emb_origin = user_emb_0.mean(dim=1)
        user_embeddings.append(user_emb_origin)

        # 选择第 0 层的知识图谱交互
        user_knowledge_emb = None

        for i in range(self.n_layer): # n_layer * [batch_size, dim]
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(user_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(user_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(user_triple_set[2][i])
            # [batch_size, dim]
            user_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            if i == 0:
                user_knowledge_emb = h_emb.mean(dim=1)
            user_embeddings.append(user_emb_i)


        # 是否使用原始的 user_id 和 item_id 过 embedding 层
        user_cross_emb = user_emb_origin

        # 交互层
        if self.args.use_cross:
            _, head_emb = self.user_cc_unit([user_cross_emb, user_knowledge_emb])
            # user_embeddings.append(self.user_embeddings)
            user_embeddings.append(head_emb)



        # 全部 user 特征拼接
        user_embeddings_concat = torch.cat(user_embeddings, axis=-1)

        return user_embeddings_concat


    def get_item_full_emb(self, items, item_triple_set):

        """ item 塔 """
        item_embeddings = []
        
        # [batch size, dim]
        item_emb_origin = self.entity_emb(items)
        item_embeddings.append(item_emb_origin)

        # 选择第 0 层的知识图谱交互
        item_knowledge_emb = None
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(item_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(item_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(item_triple_set[2][i])
            # [batch_size, dim]
            item_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            item_embeddings.append(item_emb_i)
        
            if i == 0:
                item_knowledge_emb = h_emb.mean(dim=1)

        # 是否使用原始的 user_id 和 item_id 过 embedding 层
        item_cross_emb = item_emb_origin

        # 交互层
        if self.args.use_cross:
            _, head_emb = self.item_cc_unit([item_cross_emb, item_knowledge_emb])
            item_embeddings.append(head_emb)

        item_embeddings_concat = torch.cat(item_embeddings, axis=-1)

        return item_embeddings_concat

                
    def forward(
        self,
        users: torch.LongTensor,
        items: torch.LongTensor,
        user_triple_set: list,
        item_triple_set: list,
    ):       
        
        """
            items: target-item
            user_triple_set: List 类型, [h,r,t], 其中 h r t 分别都是 List 类型, 每个元素都是 LongTensor.
        """

        # 全部 user 特征拼接
        user_embeddings_concat = self.get_user_full_emb(users, user_triple_set)

        # 全部 item 特征拼接
        item_embeddings_concat = self.get_item_full_emb(items, item_triple_set)


        # 对比学习 loss
        contrastive_loss = 0
        if self.args.use_contrastive_loss:
            contrastive_loss = self.Contrastive.semi_loss(user_embeddings_concat, item_embeddings_concat)
            contrastive_loss = torch.mean(contrastive_loss)
            contrastive_loss = self.args.contrastive_loss_weight * contrastive_loss


        """ user 知识图谱表征模块 """
        user_cos_loss = 0
        if self.args.use_kg_loss:
            h_emb_list = []
            r_emb_list = []
            t_emb_list = []

            for i in range(self.n_layer): # n_layer * [batch_size, dim]
                # [batch_size, triple_set_size, dim]
                h_emb = self.entity_emb(user_triple_set[0][i])
                # [batch_size, triple_set_size, dim]
                r_emb = self.relation_emb(user_triple_set[1][i])
                # [batch_size, triple_set_size, dim]
                t_emb = self.entity_emb(user_triple_set[2][i])

                h_emb = torch.unsqueeze(h_emb, 1)
                r_emb = torch.unsqueeze(r_emb, 1)
                t_emb = torch.unsqueeze(t_emb, 1)

                h_emb_list.append(h_emb)
                r_emb_list.append(r_emb)
                t_emb_list.append(t_emb)

            h_emb = torch.cat(h_emb_list, axis=1)
            r_emb = torch.cat(r_emb_list, axis=1)
            t_emb = torch.cat(t_emb_list, axis=1)
            
            t_emb = self.user_tail_mlp(t_emb)

            # [batch_size, dim * 2]
            head_relation_concat = torch.cat([h_emb, r_emb], -1)
            tail_pred = self.user_kge_mlp(head_relation_concat)

            # 归一化
            tail_pred = F.normalize(tail_pred, p=2, dim=-1)
            t_emb = F.normalize(t_emb, p=2, dim=-1)
            user_cos_loss = 1 - F.cosine_similarity(tail_pred, t_emb, dim=-1)
            user_cos_loss = torch.mean(user_cos_loss)
            user_cos_loss = self.args.kg_loss_weight * user_cos_loss


        """ item 知识图谱表征模块 """
        item_cos_loss = 0
        if self.args.use_kg_loss:
            h_emb_list = []
            r_emb_list = []
            t_emb_list = []

            for i in range(self.n_layer): # n_layer * [batch_size, dim]
                # [batch_size, triple_set_size, dim]
                h_emb = self.entity_emb(item_triple_set[0][i])
                # [batch_size, triple_set_size, dim]
                r_emb = self.relation_emb(item_triple_set[1][i])
                # [batch_size, triple_set_size, dim]
                t_emb = self.entity_emb(item_triple_set[2][i])

                h_emb = torch.unsqueeze(h_emb, 1)
                r_emb = torch.unsqueeze(r_emb, 1)
                t_emb = torch.unsqueeze(t_emb, 1)

                h_emb_list.append(h_emb)
                r_emb_list.append(r_emb)
                t_emb_list.append(t_emb)

            h_emb = torch.cat(h_emb_list, axis=1)
            r_emb = torch.cat(r_emb_list, axis=1)
            t_emb = torch.cat(t_emb_list, axis=1)
            
            t_emb = self.item_tail_mlp(t_emb)

            # [batch_size, dim * 2]
            head_relation_concat = torch.cat([h_emb, r_emb], -1)
            tail_pred = self.item_kge_mlp(head_relation_concat)
    
            tail_pred = F.normalize(tail_pred, p=2, dim=-1)
            t_emb = F.normalize(t_emb, p=2, dim=-1)
            item_cos_loss = 1 - F.cosine_similarity(tail_pred, t_emb, dim=-1)
            item_cos_loss = torch.mean(item_cos_loss)
            item_cos_loss = self.args.kg_loss_weight * item_cos_loss

        # 双塔预测打分
        scores = self.predict([user_embeddings_concat], [item_embeddings_concat])

        return scores, contrastive_loss, user_cos_loss, item_cos_loss
    


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

        # use_knowledge_gate
        if self.args.use_knowledge_gate:
            knowledge_gate_output = 2 * self._knowledge_gate(torch.cat((h_emb,r_emb),dim=-1))
            emb_i = knowledge_gate_output * emb_i

        # [batch_size, dim]
        emb_i = emb_i.sum(dim=1)
        return emb_i