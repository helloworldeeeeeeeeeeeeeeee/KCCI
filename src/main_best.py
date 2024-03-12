import argparse
import torch
import numpy as np
from data_loader import load_data
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='music', help='which dataset to use (music, book, movie, restaurant)')
parser.add_argument('--n_epoch', type=int, default=30, help='the number of epochs')
parser.add_argument('--batch_size', type=int, default=8192, help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=8192, help='batch size')
parser.add_argument('--n_layer', type=int, default=3, help='depth of layer')
parser.add_argument('--lr', type=float, default=0.0035, help='learning rate')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of the l2 regularization term')


parser.add_argument('--dim', type=int, default=128, help='dimension of entity and relation embeddings')
parser.add_argument('--user_triple_set_size', type=int, default=32, help='the number of triples in triple set of user')
parser.add_argument('--item_triple_set_size', type=int, default=16, help='the number of triples in triple set of item')
parser.add_argument('--agg', type=str, default='concat', help='the type of aggregator (sum, pool, concat)')


parser.add_argument('--temperature', type=float, default=15, help='temperature')
parser.add_argument('--random_seed', type=int, default=42, help='random_seed')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether using gpu or cpu')
parser.add_argument('--show_topk', type=bool, default=False, help='whether showing topk or not')
parser.add_argument('--random_flag', type=bool, default=False, help='whether using random seed or not')
parser.add_argument('--use_raw_id_emb', type=bool, default=False, help='whether use_raw_id_emb')
parser.add_argument('--use_user_kg_loss', type=bool, default=True, help='whether use_user_kg_loss')
parser.add_argument('--use_item_kg_loss', type=bool, default=True, help='whether use_item_kg_loss')




args = parser.parse_args()

print(" --------------  打印所有的参数 ----------------")
print(args)
print(" ---------------------------------------------")

def set_random_seed(torch_seed):
    np.random.seed(torch_seed)                  
    torch.manual_seed(torch_seed)       
    torch.cuda.manual_seed(torch_seed)      
    torch.cuda.manual_seed_all(torch_seed)  

if not args.random_flag:
    set_random_seed(args.random_seed)
    
data_info = load_data(args)
train(args, data_info)
    