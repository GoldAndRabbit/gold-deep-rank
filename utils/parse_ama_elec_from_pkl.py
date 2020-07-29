import pickle
import pandas as pd
from tqdm import tqdm

PATH = '/census_data/'

def list_to_str(a_list):
    return ",".join(list(map(str, a_list)))


def parse():
    with open(PATH + '/dataset.pkl', 'rb') as f:
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
        df_train['user_id'] = df_train['user_id'].astype(str)
        df_train['seq'] = train_seq
        df_train['item_id'] = train_tar_item
        df_train['item_id'] = df_train['item_id'].astype(str)
        df_train['seq_cate'] = train_seq_cate
        df_train['item_cate'] = train_tar_cate
        df_train['item_cate'] = df_train['item_cate'].astype(str)
        df_train['label'] = train_label
        df_train['label'] = df_train['label'].astype(str)

        df_test['user_id'] = test_user_id
        df_test['user_id'] = df_test['user_id'].astype(str)
        df_test['seq'] = test_seq
        df_test['item_id'] = test_tar_item
        df_test['item_id'] = df_test['item_id'].astype(str)
        df_test['seq_cate'] = test_seq_cate
        df_test['item_cate'] = test_tar_cate
        df_test['item_cate'] = df_test['item_cate'].astype(str)
        df_test['label'] = test_label
        df_test['label'] = df_test['label'].astype(str)
        df_test = df_test.sample(frac=1)

        df_train.to_csv(PATH + 'ama_ele_train.csv', index=False)
        df_test.to_csv(PATH + 'ama_ele_test.csv', index=False)


def padding_data_to_csv():
    def _padding_up_to_10(x):
        data = x.split(',')
        if len(data) >= 10:
            tmp = data[-10:]
            return ','.join(tmp)
        else:
            gap = 10 - len(data)
            pad = [''] * gap
            tmp = pad + data
            return ','.join(tmp)
    df_train = pd.read_csv(PATH + 'ama_ele_train.csv')
    df_test = pd.read_csv(PATH + 'ama_ele_test.csv')
    df_train['seq'] = df_train['seq'].apply(_padding_up_to_10)
    df_train['seq_cate'] = df_train['seq_cate'].apply(_padding_up_to_10)
    df_test['seq'] = df_test['seq'].apply(_padding_up_to_10)
    df_test['seq_cate'] = df_test['seq_cate'].apply(_padding_up_to_10)
    df_train.to_csv(PATH + 'ama_ele_train_pad.csv', index=False)
    df_test.to_csv(PATH + 'ama_ele_test_pad.csv', index=False)


if __name__ == '__main__':
    # parse()
    padding_data_to_csv()
