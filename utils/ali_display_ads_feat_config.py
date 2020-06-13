import tensorflow.feature_column as fc
import pandas as pd
import logging


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

def build_ali_display_ads_wide_columns():
    numeric_feat = ['price']
    category_feat = [x for x in ALI_DISPLAY_ADS_COLUMNS if x not in numeric_feat and x != 'clk']
    hash_bucket_config = {
        'userid':       50000000,
        'adgroup_id':   10000000,
        'cate_id':      100000,
        'pid':          1000000,
        'campain_id':   1000000,
    }
    base_columns = []
    for feat in category_feat:
        base_columns.append(fc.indicator_column(fc.categorical_column_with_hash_bucket(feat, hash_bucket_size=100000)))

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
    hash_bucket_config = {
        'userid':       200000000,
        'adgroup_id':   50000000,
        'cate_id':      100000,
        'pid':          1000000,
        'campain_id':   1000000,
    }
    feature_columns = []
    for feat in category_feat:
        feature_columns.append(fc.embedding_column(
            fc.categorical_column_with_hash_bucket(feat, hash_bucket_size=hash_bucket_config.get(feat, 100000)), dimension=emb_dims))
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
    PATH = '/media/psdz/hdd/Download/ali_display_ads/'
    logging.info('finish loading...')
    ad_feature = pd.read_csv(PATH + 'ad_feature.csv')
    user_feature = pd.read_csv(PATH + 'user_profile.csv')
    raw_sample = pd.read_csv(PATH + 'raw_sample.csv')
    logging.info('finish mapping date...')
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
    logging.info('finish merging...')
    raw_sample = pd.merge(left=raw_sample, right=user_feature, on='userid', how='left')
    raw_sample = pd.merge(left=raw_sample, right=ad_feature, on='adgroup_id', how='left')
    raw_sample['new_user_class_level'] = raw_sample['new_user_class_level ']
    raw_sample['customer_id'] = raw_sample['customer']
    raw_sample['price'] = raw_sample['price'].map(lambda x: int(x))
    # behavior_log may consume large memory.
    # behavoir_log = pd.read_csv(PATH + 'behavior_log.csv')
    df_train = raw_sample.loc[raw_sample['time_stamp']<='20170512']
    df_test = raw_sample.loc[raw_sample['time_stamp']=='20170513']
    df_train = df_train[ALI_DISPLAY_ADS_COLUMNS]
    df_test = df_test[ALI_DISPLAY_ADS_COLUMNS]
    df_train_sample = df_train.head(20000)
    df_test_sample = df_test.head(20000)
    # NOTE: writing csv file without header.
    logging.info('finish writing...')
    df_train.to_csv(PATH + 'train_log.csv', index=False, header=None)
    df_test.to_csv(PATH + 'test_log.csv', index=False, header=None)
    df_train_sample.to_csv(PATH + 'train_log_sample.csv', index=False, header=None)
    df_test_sample.to_csv(PATH + 'test_log_sample.csv', index=False, header=None)


def test_ali_display_ads_dataset():
    PATH = '/media/psdz/hdd/Download/ali_display_ads/'
    a = pd.read_csv(PATH + 'train_log_sample.csv', names=ALI_DISPLAY_ADS_COLUMNS)
    pos_sample_ratio = a[['clk', 'userid']].groupby(['clk'],as_index=False).count()
    print(a.info())


if __name__ == '__main__':
    generate_ali_display_ads_dataset()
