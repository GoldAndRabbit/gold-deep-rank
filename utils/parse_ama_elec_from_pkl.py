import os
import pickle
from collections import Counter
from tqdm import tqdm
import pandas as pd

path = os.getcwd()


def list_to_str(a_list):
    return ",".join(list(map(str, a_list)))


def parse_dump_data():
    with open('../toy_data/ama_elec.pkl', 'rb') as f:
        # train_set example (189526, [18294, 26271, 14004, 24418, 15043, 1581, 7901], 41583, 0)
        # test_Set example (98577, [299, 5127, 6140, 14805, 15916, 20120], (17961, 29586))
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count = pickle.load(f) # 192403, 63001, 801
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
        df_train['seq'] = train_seq
        df_train['item_id'] = train_tar_item
        df_train['seq_cate'] = train_seq_cate
        df_train['item_cate'] = train_tar_cate
        df_train['label'] = train_label
        df_test['user_id'] = test_user_id
        df_test['seq'] = test_seq
        df_test['item_id'] = test_tar_item
        df_test['seq_cate'] = test_seq_cate
        df_test['item_cate'] = test_tar_cate
        df_test['label'] = test_label
        df_test = df_test.sample(frac=1)

        df_train.to_csv(path.replace('utils', '/toy_data/ama_ele_train.csv'), index=False)
        df_test.to_csv(path.replace('utils', '/toy_data/ama_ele_test.csv'), index=False)


def make_fixed_len_feature():
    df_train = pd.read_csv(path.replace('utils', '/toy_data/ama_ele_train.csv'))
    df_test = pd.read_csv(path.replace('utils', '/toy_data/ama_ele_train.csv'))
    seq = df_train['seq'].tolist()
    seq_cate = df_train['seq_cate'].to_list()
    pad_seq = []
    pad_seq_cate = []
    for (x, y) in zip(seq, seq_cate):
        x = x.split(',')
        y = y.split(',')
        if len(x) <= 20:
            gap = 20 - len(x)
            tmp_x = gap * ['-1'] + x
            tmp_y = gap * ['-1'] + y
        else:
            tmp_x = x[-20:]
            tmp_y = y[-20:]
        pad_seq.append(list_to_str(tmp_x))
        pad_seq_cate.append(list_to_str(tmp_y))
    df_train['seq'] = pad_seq
    df_train['seq_cate'] = pad_seq_cate

    seq = df_test['seq'].tolist()
    seq_cate = df_test['seq_cate'].to_list()
    pad_seq = []
    pad_seq_cate = []
    for (x, y) in zip(seq, seq_cate):
        x = x.split(',')
        y = y.split(',')
        if len(x) <= 20:
            gap = 20 - len(x)
            tmp_x = gap * ['-1'] + x
            tmp_y = gap * ['-1'] + y
        else:
            tmp_x = x[-20:]
            tmp_y = y[-20:]
        pad_seq.append(list_to_str(tmp_x))
        pad_seq_cate.append(list_to_str(tmp_y))
    df_test['seq'] = pad_seq
    df_test['seq_cate'] = pad_seq_cate

    df_train.to_csv(path.replace('utils', '/toy_data/ama_ele_train_pad.csv'), index=False)
    df_test.to_csv(path.replace('utils', '/toy_data/ama_ele_test_pad.csv'), index=False)

if __name__ == '__main__':
    parse_dump_data()
    make_fixed_len_feature()