import collections
import os
import numpy as np
import logging

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


def load_data(args):
    logging.info("================== preparing data ===================")
    train_data, eval_data, test_data, user_init_entity_set, item_init_entity_set, n_user, n_item = load_rating(args)
    n_entity, n_relation, kg = load_kg(args)
    logging.info("contructing users' kg triple sets ...")
    user_triple_sets = kg_propagation(args, kg, user_init_entity_set, args.user_triple_set_size, True)
    logging.info("contructing items' kg triple sets ...")
    item_triple_sets = kg_propagation(args, kg, item_init_entity_set, args.item_triple_set_size, False)

    print("n_user: ", n_user, "n_item: ", n_item, "n_entity: ", n_entity, "n_relation: ", n_relation)


    return train_data, eval_data, test_data, n_entity, n_relation, user_triple_sets, item_triple_sets, n_user, n_item




def get_item_to_triples(item_id, kg):
    """
        输入 item_id, 获得与之相关的 triple
    """

    h = []
    t = []
    r = []

    for tail_and_relation in kg[item_id]:
        h.append(item_id)
        t.append(tail_and_relation[0])
        r.append(tail_and_relation[1])

    return [h, t, r]


def load_rating(args):
    """ 
        加载数据集, 读取的数据格式为 user_id, item_id, label
        通过 dataset_split 生成 train eval test 数据
    """
    rating_file = '../data/' + args.dataset + '/ratings_final'
    logging.info("load rating file: %s.npy", rating_file)
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)
    return dataset_split(rating_np)


def dataset_split(rating_np):
    """
        将原始数据集进行划分, 默认为 train:eval:test = 6:2:2
        返回 3 个数据集 和 经过 u2i、i2u2i 传播过后的 user_init_entity、item_init_entiy, 里面存的是 item 而不是实体
    """
    logging.info("splitting dataset to 6:2:2 ...")
    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]
    
    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    
    user_init_entity_set, item_init_entity_set = collaboration_propagation(rating_np, train_indices)
    
    train_indices = [i for i in train_indices if rating_np[i][0] in user_init_entity_set.keys()]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_init_entity_set.keys()]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_init_entity_set.keys()]
    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    # 获得全部的 user 和 item
    all_user = rating_np[:, 0]
    all_item = rating_np[:, 1]
    n_user = len(set(list(all_user)))
    n_item = len(set(list(all_item)))


    return train_data, eval_data, test_data, user_init_entity_set, item_init_entity_set, n_user, n_item
    
    
def collaboration_propagation(rating_np, train_indices):
    """ 
        经过 u2i、i2u2i 传播后的 user_init_entity、item_init_entiy, 里面存的是 item 而不是实体
        1. 获得 user 历史交互的 item 
        2. 获得 item 的邻居 item
    """
    logging.info("contructing users' initial entity set ...")
    user_history_item_dict = dict()
    item_history_user_dict = dict()
    item_neighbor_item_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_item_dict:
                user_history_item_dict[user] = []
            user_history_item_dict[user].append(item)
            if item not in item_history_user_dict:
                item_history_user_dict[item] = []
            item_history_user_dict[item].append(user)
        
    logging.info("contructing items' initial entity set ...")
    for item in item_history_user_dict.keys():
        item_nerghbor_item = []
        for user in item_history_user_dict[item]:
            item_nerghbor_item = np.concatenate((item_nerghbor_item, user_history_item_dict[user]))
        item_neighbor_item_dict[item] = list(set(item_nerghbor_item))

    item_list = set(rating_np[:, 1])
    for item in item_list:
        if item not in item_neighbor_item_dict:
            item_neighbor_item_dict[item] = [item]
    return user_history_item_dict, item_neighbor_item_dict


def load_kg(args):
    """ 
        加载知识图谱数据
        返回 entity 数目, relation 数量 和 kg 对象(dict 类型, key 是 head, value 是 List, 每个元素为 [tail, relation])
    """


    kg_file = '../data/' + args.dataset + '/kg_final'
    logging.info("loading kg file: %s.npy", kg_file)
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)
    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    kg = construct_kg(kg_np)
    return n_entity, n_relation, kg


def construct_kg(kg_np):
    """ 基于 head, relation, tail 格式的知识图谱数据构建 dict 类型的知识图谱对象  """

    logging.info("constructing knowledge graph ...")
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg


def kg_propagation(args, kg, init_entity_set, set_size, is_user):
    """
        根据传播后的 init_entity_set 变成 kg 训练所需要的 dict: obj-三元组 的形式, obj 为 user_id 或者 item_id
            kg: dict 类型的知识图谱对象
            init_entity_set: 经过 u2i 或者 i2u2i 传播后的 item 集合
            set_size: user_triple_set_size 参数
            is_user: 是否是 user

        此处只使用到了 user_init_entity_set, 所以此处的 obj 都是 user_id
    """
    # triple_sets: [n_obj][n_layer](h,r,t)x[set_size] 
    triple_sets = collections.defaultdict(list)
    for obj in init_entity_set.keys():
        if is_user and args.n_layer == 0:
            n_layer = 1
        else:
            n_layer = args.n_layer
        for l in range(n_layer):
            h,r,t = [],[],[]
            if l == 0:
                entities = init_entity_set[obj]
            else:
                entities = triple_sets[obj][-1][2]

            for entity in entities:
                for tail_and_relation in kg[entity]:
                    h.append(entity)
                    t.append(tail_and_relation[0])
                    r.append(tail_and_relation[1])
                    
            if len(h) == 0:
                triple_sets[obj].append(triple_sets[obj][-1])
            else:
                indices = np.random.choice(len(h), size=set_size, replace= (len(h) < set_size))
                h = [h[i] for i in indices]
                r = [r[i] for i in indices]
                t = [t[i] for i in indices]
                triple_sets[obj].append((h, r, t))
    return triple_sets
