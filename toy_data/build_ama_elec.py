import pickle
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import numpy as np


def list_to_str(a_list):
    return ",".join(list(map(str, a_list)))

def parse_dump_data():
    with open('/home/psdz/dl/gold_deep_ctr/din/dataset.pkl', 'rb') as f:
        # train_set example (189526, [18294, 26271, 14004, 24418, 15043, 1581, 7901], 41583, 0)
        # test_Set example (98577, [299, 5127, 6140, 14805, 15916, 20120], (17961, 29586))
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count = pickle.load(f) # 192403, 63001, 801
        from collections import Counter
        c = Counter(cate_list)
        # 801 cate

        train_user_id, train_seq, train_tar_item, train_label, train_seq_cate, train_tar_cate = [], [], [], [], [], []
        test_user_id, test_seq, test_tar_item, test_label, test_seq_cate, test_tar_cate = [], [], [], [], [], []

        item2cate_map = {}
        for i in range(len(cate_list)):
            item2cate_map[i] = cate_list[i]

        for x in tqdm(train_set):
            train_user_id.append(x[0])
            train_seq.append(list_to_str(x[1]))
            train_tar_item.append(x[2])
            train_label.append(x[3])
            train_seq_cate.append(list_to_str([item2cate_map[x] for x in x[1]]))
            train_tar_cate.append(item2cate_map[x[2]])

        for x in tqdm(test_set):
            test_user_id.append(x[0])
            test_seq.append(list_to_str(x[1]))
            test_tar_item.append(x[2][0])
            test_label.append(1)
            test_seq_cate.append(list_to_str([item2cate_map[x] for x in x[1]]))
            test_tar_cate.append(item2cate_map[x[2][0]])

        for x in tqdm(test_set):
            test_user_id.append(x[0])
            test_seq.append(list_to_str(x[1]))
            test_tar_item.append(x[2][1])
            test_label.append(0)
            test_seq_cate.append(list_to_str([item2cate_map[x] for x in x[1]]))
            test_tar_cate.append(item2cate_map[x[2][1]])

        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        df_train['user_id'] = train_user_id
        # df_train['user_id'] = df_train['user_id'].astype(str)
        df_train['seq'] = train_seq
        df_train['item_id'] = train_tar_item
        # df_train['item_id'] = df_train['item_id'].astype(str)
        df_train['seq_cate'] = train_seq_cate
        df_train['item_cate'] = train_tar_cate
        # df_train['item_cate'] = df_train['item_cate'].astype(str)
        df_train['label'] = train_label
        # df_train['label'] = df_train['label'].astype(str)

        df_test['user_id'] = test_user_id
        # df_test['user_id'] = df_test['user_id'].astype(str)
        df_test['seq'] = test_seq
        df_test['item_id'] = test_tar_item
        # df_test['item_id'] = df_test['item_id'].astype(str)
        df_test['seq_cate'] = test_seq_cate
        df_test['item_cate'] = test_tar_cate
        # df_test['item_cate'] = df_test['item_cate'].astype(str)
        df_test['label'] = test_label
        # df_test['label'] = df_test['label'].astype(str)
        df_test = df_test.sample(frac=1)

        # df_train.to_csv('./ama_ele_train.csv', index=False, header=None)
        # df_test.to_csv('./ama_ele_test.csv', index=False, header=None)

        df_train.to_csv('./ama_ele_train.csv', index=False)
        df_test.to_csv('./ama_ele_test.csv', index=False)


def dump_dict():
    df_train = pd.read_csv('./ama_ele_train.csv')
    # df_train = df_train.head(10)
    df_train_seq = df_train[['seq','seq_cate']]
    df_train_non_seq = df_train.drop(['seq','seq_cate'],axis=1)
    train_non_seq_dict = df_train_non_seq.to_dict(orient='list')
    train_seq_dict = df_train_seq.to_dict(orient='list')
    for k, v in train_non_seq_dict.items():
        train_non_seq_dict[k] = np.array(v)
    train_dict = {}
    for k, v in tqdm(train_seq_dict.items()):
        train_dict[k] = []
        for seq_str in v:
            train_dict[k].append(list(map(int, seq_str.split(','))))
    for k, v in train_dict.items():
        train_dict[k] = np.array(v)
    train_dict.update(train_non_seq_dict)

    df_test = pd.read_csv('./ama_ele_test.csv')
    # df_test = df_test.head(10)
    df_test_seq = df_test[['seq','seq_cate']]
    df_test_non_seq = df_test.drop(['seq','seq_cate'],axis=1)
    test_non_seq_dict = df_test_non_seq.to_dict(orient='list')
    test_seq_dict = df_test_seq.to_dict(orient='list')
    for k, v in test_non_seq_dict.items():
        test_non_seq_dict[k] = np.array(v)
    test_dict = {}
    for k, v in tqdm(test_seq_dict.items()):
        test_dict[k] = []
        for seq_str in v:
            test_dict[k].append(list(map(int, seq_str.split(','))))
    for k, v in test_dict.items():
        test_dict[k] = np.array(v)
    test_dict.update(test_non_seq_dict)

    pickle.dump(train_dict, open('ama_ele_train_dict.pkl', 'wb'))
    pickle.dump(test_dict, open('ama_ele_test_dict.pkl', 'wb'))

def dump_dict_10():
    df_train = pd.read_csv('./ama_ele_train.csv')
    df_train = df_train.head(10)
    df_train_seq = df_train[['seq','seq_cate']]
    df_train_non_seq = df_train.drop(['seq','seq_cate'],axis=1)
    train_non_seq_dict = df_train_non_seq.to_dict(orient='list')
    train_seq_dict = df_train_seq.to_dict(orient='list')
    for k, v in train_non_seq_dict.items():
        train_non_seq_dict[k] = np.array(v)
    train_dict = {}
    for k, v in tqdm(train_seq_dict.items()):
        train_dict[k] = []
        for seq_str in v:
            train_dict[k].append(list(map(int, seq_str.split(','))))
        train_dict[k] = np.array(train_dict[k])
    # for k, v in train_dict.items():
    #     v_arr = np.array(v)
    train_dict.update(train_non_seq_dict)
    pickle.dump(train_dict, open('ama_ele_train_dict_10.pkl', 'wb'))

def load_dict():
    a = pickle.load(open('ama_ele_train_dict.pkl', 'rb'))
    b = pickle.load(open('ama_ele_test_dict.pkl', 'rb'))
    pprint(a)
    pprint(b)

def load_dict_10():
    a = pickle.load(open('ama_ele_train_dict_10.pkl', 'rb'))
    pprint(a)

def make_data():
    train_df = pd.read_csv('~/repos/Gold_DeepCTR_Tensorflow/toy_data/ama_ele_train.csv')
    test_df = pd.read_csv('~/repos/Gold_DeepCTR_Tensorflow/toy_data/ama_ele_test.csv')

    train_df.to_csv('ama_ele_train_v0.csv', sep='\t', index=False, header=None)
    test_df.to_csv('ama_ele_test_v0.csv', sep='\t', index=False, header=None)
    print(train_df.head(5))

if __name__ == '__main__':
    parse_dump_data()
    dump_dict_10()
    load_dict_10()
    # make_data()

