import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def row_column_shuffle(embedding):
    """ 输入一个 batch 的数据, 用于随机打乱的函数 """
    
    corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
    corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
    return corrupted_embedding

def emb_score(x1, x2):
    """ 计算两个向量点乘 """

    return torch.sum(torch.mul(x1, x2), 1)


def co_loss(h1, h2):
    pos = emb_score(h1, h2)  # 正样本是 两个视角的交互
    neg1 = emb_score(h1, row_column_shuffle(h2))  # 负样本是 打乱交互
    one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1)
    sub_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg1))))
    return sub_loss



class Contrast(nn.Module):
    def __init__(self, temperature=1):
        super(Contrast, self).__init__()
        self.temperature = temperature

    def pair_sim(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        sim = torch.mm(z1, z2.t())

        return sim


    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.temperature)
        self_sim = f(self.pair_sim(z1, z1))
        between_sim = f(self.pair_sim(z1, z2))

        # 实现 batch 内其他为负样本
        semi_loss = -torch.log(
            between_sim.diag() /
            (self_sim.sum(1) + between_sim.sum(1) - self_sim.diag()))

        return semi_loss




class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0, chnl=8):
        super(Dense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = nn.ReLU()
        self.drop_layer = nn.Dropout(p=self.dropout) # Pytorch drop: ratio to zeroed
        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, inputs):
        x = self.drop_layer(inputs)
        output = self.fc(x)
        return self.act(output)


class Bias(nn.Module):
    def __init__(self, dim):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        x = x + self.bias
        return x


class CrossCompressUnit(nn.Module):
    def __init__(self, dim):
        super(CrossCompressUnit, self).__init__()
        self.dim = dim
        self.fc_vv = nn.Linear(dim, 1, bias=False)
        self.fc_ev = nn.Linear(dim, 1, bias=False)
        self.fc_ve = nn.Linear(dim, 1, bias=False)
        self.fc_ee = nn.Linear(dim, 1, bias=False)

        self.bias_v = Bias(dim)
        self.bias_e = Bias(dim)

        # self.fc_v = nn.Linear(dim, dim)
        # self.fc_e = nn.Linear(dim, dim)

    def forward(self, inputs):
        v, e = inputs

        # [batch_size, dim, 1], [batch_size, 1, dim]
        v = torch.unsqueeze(v, 2)
        e = torch.unsqueeze(e, 1)

        # [batch_size, dim, dim]
        c_matrix = torch.matmul(v, e)
        c_matrix_transpose = c_matrix.permute(0,2,1)

        # [batch_size * dim, dim]
        c_matrix = c_matrix.view(-1, self.dim)
        c_matrix_transpose = c_matrix_transpose.contiguous().view(-1, self.dim)

        # [batch_size, dim]
        v_intermediate = self.fc_vv(c_matrix) + self.fc_ev(c_matrix_transpose)
        e_intermediate = self.fc_ve(c_matrix) + self.fc_ee(c_matrix_transpose)
        v_intermediate = v_intermediate.view(-1, self.dim)
        e_intermediate = e_intermediate.view(-1, self.dim)

        v_output = self.bias_v(v_intermediate)
        e_output = self.bias_e(e_intermediate)


        return v_output, e_output


