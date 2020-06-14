import logging
import pandas as pd
import tensorflow.feature_column as fc

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

PATH = '/media/psdz/hdd/Download/ali_display_ads/'

ALI_DISPLAY_ADS_COLUMNS = [
    # user feat: 9
    'userid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level',
    # ad feat: 7
    'adgroup_id', 'cate_id', 'campaign_id', 'customer_id', 'brand', 'pid', 'price',
    # label
    'clk'
]

ALI_DISPLAY_ADS_COLUMN_DEFAULTS = [
    [''],[''],[''],[''],[''],[''],[''],[''],[''],
    [''],[''],[''],[''],[''],[''], [-1],
    [0]
]

HASH_BUCKET_CONFIG = {
    'adgroup_id': 846811,
    'age_level': 8,
    'brand': 99815,
    'campaign_id': 423436,
    'cate_id': 6769,
    'cms_group_id': 14,
    'cms_segid': 98,
    'customer_id': 255875,
    'final_gender_code': 3,
    'new_user_class_level': 5,
    'occupation': 3,
    'pid': 2,
    'pvalue_level': 4,
    'shopping_level': 4,
    'userid': 1141729
}

def build_ali_display_ads_wide_columns():
    numeric_feat = ['price']
    category_feat = [x for x in ALI_DISPLAY_ADS_COLUMNS if x not in numeric_feat and x != 'clk']

    base_columns = []
    for feat in category_feat:
        cur_hash_size = 1000
        if HASH_BUCKET_CONFIG[feat] > 1000:
            cur_hash_size = HASH_BUCKET_CONFIG[feat] + 100000
        base_columns.append(fc.indicator_column(fc.categorical_column_with_hash_bucket(feat, hash_bucket_size=cur_hash_size)))
    cross_columns = [
        # fc.indicator_column(fc.crossed_column(["education", "occupation"], hash_bucket_size=1000)),
        # fc.indicator_column(fc.crossed_column(["native_country", "occupation"], hash_bucket_size=1000)),
    ]
    feature_columns = base_columns + cross_columns
    feat_field_size = len(feature_columns)
    return feature_columns, feat_field_size

def build_ali_display_ads_columns(emb_dims=8):
    numeric_feat = ['price']
    category_feat = [x for x in ALI_DISPLAY_ADS_COLUMNS if x not in numeric_feat and x != 'clk']
    feature_columns = []
    for feat in category_feat:
        cur_hash_size = 1000
        if HASH_BUCKET_CONFIG[feat] > 1000:
            cur_hash_size = HASH_BUCKET_CONFIG[feat] + 100000
        feature_columns.append(fc.embedding_column(fc.categorical_column_with_hash_bucket(feat, hash_bucket_size=cur_hash_size), dimension=emb_dims))
    for feat in numeric_feat:
        feature_columns.append(fc.numeric_column(feat))
    feat_field_size = len(feature_columns)
    return feature_columns, feat_field_size


def unixstamp2date(x):
    import datetime
    customed_format = '%Y%m%d%H%M%S'
    date_time = datetime.datetime.fromtimestamp(x)
    time_str = date_time.strftime(customed_format)
    time_str = time_str[:8]
    return time_str


def generate_ali_display_ads_dataset():
    logging.info('finish loading raw data...')
    ad_feature = pd.read_csv(PATH + 'ad_feature.csv')
    user_feature = pd.read_csv(PATH + 'user_profile.csv')
    raw_sample = pd.read_csv(PATH + 'raw_sample.csv')
    logging.info('finish transforming unix date...')
    raw_sample['time_stamp'] = raw_sample['time_stamp'].apply(unixstamp2date)
    raw_sample = raw_sample.drop(['nonclk'], axis=1)
    '''
    # pos_sample_ratio = raw_sample[['clk', 'time_stamp']].groupby(['clk'], as_index=False).count()
    0: 25191905
    1: 1366056
    pos_sample_ratio: 0.054 
    '''
    raw_sample['userid'] = raw_sample['user']
    raw_sample = raw_sample[['userid', 'adgroup_id', 'pid', 'time_stamp', 'clk']]
    logging.info('finish merging basic feat...')
    raw_sample = pd.merge(left=raw_sample, right=user_feature, on='userid', how='left')
    raw_sample = pd.merge(left=raw_sample, right=ad_feature, on='adgroup_id', how='left')
    raw_sample['new_user_class_level'] = raw_sample['new_user_class_level ']
    raw_sample['customer_id'] = raw_sample['customer']
    raw_sample['price'] = raw_sample['price'].map(lambda x: int(x))
    raw_sample.to_csv(PATH + 'day_sample.csv', index=False)
    df_train = raw_sample.loc[raw_sample['time_stamp']<='20170512']
    df_test = raw_sample.loc[raw_sample['time_stamp']=='20170513']
    df_train = df_train[ALI_DISPLAY_ADS_COLUMNS]
    df_test = df_test[ALI_DISPLAY_ADS_COLUMNS]
    logging.info('finish shuffling data...')
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    df_train_sample = df_train.head(20000)
    df_test_sample = df_test.head(20000)
    # NOTE: writing csv file without header.
    logging.info('finish writing...')
    df_train.to_csv(PATH + 'train_log.csv', index=False, header=None)
    df_test.to_csv(PATH + 'test_log.csv', index=False, header=None)
    df_train_sample.to_csv(PATH + 'train_log_sample.csv', index=False, header=None)
    df_test_sample.to_csv(PATH + 'test_log_sample.csv', index=False, header=None)


def feat_unique_count():
    logging.info('finish counting...')
    day_sample = pd.read_csv(PATH + 'day_sample.csv')
    feat_cnt = {}
    for feat in ALI_DISPLAY_ADS_COLUMNS:
        if feat != 'price' and feat != 'clk':
            tmp = day_sample[feat].unique()
            feat_cnt[feat] = len(tmp)
            print('%s: %d' % (feat, len(tmp)))
    '''
    {'adgroup_id': 846811,
     'age_level': 8,
     'brand': 99815,
     'campaign_id': 423436,
     'cate_id': 6769,
     'cms_group_id': 14,
     'cms_segid': 98,
     'customer_id': 255875,
     'final_gender_code': 3,
     'new_user_class_level': 5,
     'occupation': 3,
     'pid': 2,
     'pvalue_level': 4,
     'shopping_level': 4,
     'userid': 1141729}
    '''
    return feat_cnt


def generate_stat_feature():
    # behavior_log consume large memory.
    behavoir_log = pd.read_csv(PATH + 'behavior_log.csv')


def test_ali_display_ads_dataset():
    PATH = '/media/psdz/hdd/Download/ali_display_ads/'
    a = pd.read_csv(PATH + 'train_log_sample.csv', names=ALI_DISPLAY_ADS_COLUMNS)
    pos_sample_ratio = a[['clk', 'userid']].groupby(['clk'],as_index=False).count()
    logging.info(a.info())


if __name__ == '__main__':
    # generate_ali_display_ads_dataset()
    feat_unique_count()

