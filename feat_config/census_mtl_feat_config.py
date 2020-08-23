import tensorflow.feature_column as fc
import numpy as np
import pandas as pd

CENSUS_COLUMNS = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','gender','capital_gain','capital_loss','hours_per_week','native_country','income_bracket']
CENSUS_COLUMN_DEFAULTS = [[-1],[''],[-1],[''],[-1],[''],[''],[''],[''],[''],[-1],[-1],[-1],[''],['']]
SEQ_COLUMNS = ['user_id', 'seq', 'item_id', 'seq_cate', 'item_cate', 'label']
SEQ_COLUMNS_DEFAULTS = [[''], [''], [''], [''], [''], ['0']]

def get_census_numeric_feat_range():
    train = pd.read_csv('./census_data/adult.data',header=None,names=CENSUS_COLUMNS)[['age','education_num','capital_gain','capital_loss','hours_per_week']]
    test = pd.read_csv('./census_data/adult.test',header=None,names=CENSUS_COLUMNS)[['age','education_num','capital_gain','capital_loss','hours_per_week']]
    total = pd.concat([train, test], axis=0)
    n_range = dict()
    n_range['age'] = (total['age'].min(), total['age'].max())
    n_range['education_num'] = (total['education_num'].min(), total['education_num'].max())
    n_range['capital_gain'] = (total['capital_gain'].min(), total['capital_gain'].max())
    n_range['capital_loss'] = (total['capital_loss'].min(), total['capital_loss'].max())
    n_range['hours_per_week'] = (total['hours_per_week'].min(), total['hours_per_week'].max())
    return n_range


def build_mtl_census_emb_columns():
    n_range = get_census_numeric_feat_range()
    feature_columns = [
        # numeric feature embedding
        fc.embedding_column(fc.bucketized_column(fc.numeric_column('age'), boundaries=list(np.linspace(n_range['age'][0], n_range['age'][1], 1000))), dimension=32),
        fc.embedding_column(fc.bucketized_column(fc.numeric_column('education_num'), boundaries=list(np.linspace(n_range['education_num'][0], n_range['education_num'][1], 1000))), dimension=32),
        fc.embedding_column(fc.bucketized_column(fc.numeric_column('capital_gain'), boundaries=list(np.linspace(n_range['capital_gain'][0], n_range['capital_gain'][1], 1000))), dimension=32),
        fc.embedding_column(fc.bucketized_column(fc.numeric_column('capital_loss'), boundaries=list(np.linspace(n_range['capital_loss'][0], n_range['capital_loss'][1], 1000))), dimension=32),
        fc.embedding_column(fc.bucketized_column(fc.numeric_column('hours_per_week'),boundaries=list(np.linspace(n_range['hours_per_week'][0], n_range['hours_per_week'][1], 1000))), dimension=32),
        # category feature embedding
        fc.embedding_column(fc.categorical_column_with_hash_bucket('gender', hash_bucket_size=1000), dimension=32),
        fc.embedding_column(fc.categorical_column_with_hash_bucket('education', hash_bucket_size=1000), dimension=32),
        # remove marital_status
        # fc.embedding_column(fc.categorical_column_with_hash_bucket('marital_status', hash_bucket_size=1000), dimension=32),
        fc.embedding_column(fc.categorical_column_with_hash_bucket('relationship', hash_bucket_size=1000), dimension=32),
        fc.embedding_column(fc.categorical_column_with_hash_bucket('workclass', hash_bucket_size=1000), dimension=32),
        fc.embedding_column(fc.categorical_column_with_hash_bucket('native_country', hash_bucket_size=1000), dimension=32),
        fc.embedding_column(fc.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000), dimension=32)
    ]
    feat_field_size = len(feature_columns)
    return feature_columns, feat_field_size

