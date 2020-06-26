import tensorflow.feature_column as fc
import pandas as pd
import numpy as np

CENSUS_COLUMNS = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','gender','capital_gain','capital_loss','hours_per_week','native_country','income_bracket']
CENSUS_COLUMN_DEFAULTS = [[-1],[''],[-1],[''],[-1],[''],[''],[''],[''],[''],[-1],[-1],[-1],[''],['']]
SEQ_COLUMNS = ['user_id', 'seq', 'item_id', 'seq_cate', 'item_cate', 'label']
SEQ_COLUMNS_DEFAULTS = [[''], [''], [''], [''], [''], ['0']]

WDL_CONFIG = {}
WDL_CONFIG['deep_emb_cols'] = ['gender','education','relationship','marital_status','workclass','native_country','occupation']
WDL_CONFIG['deep_bucket_emb_cols'] = ['age','education_num','capital_gain','capital_loss','hours_per_week']
WDL_CONFIG['wide_muti_hot_cols'] = WDL_CONFIG['deep_emb_cols']
WDL_CONFIG['wide_bucket_cols'] = WDL_CONFIG['deep_bucket_emb_cols']
WDL_CONFIG['wide_cross_cols'] = [
    ('education','occupation'),
    ('native_country','occupation'),
    ('gender','occupation')
]


def build_ama_ele_columns():
    feature_columns = [
        fc.embedding_column(fc.categorical_column_with_hash_bucket('user_id',   hash_bucket_size=200000), dimension=32),
        fc.embedding_column(fc.categorical_column_with_hash_bucket('item_id',   hash_bucket_size=70000),  dimension=32),
        fc.embedding_column(fc.categorical_column_with_hash_bucket('item_cate', hash_bucket_size=1000),   dimension=32),
    ]
    feat_field_size = len(feature_columns)
    return feature_columns, feat_field_size


def build_census_feat_columns(emb_dim=8):
    def _get_numeric_feat_range():
        train = pd.read_csv('toy_data/adult.data', header=None, names=CENSUS_COLUMNS)[WDL_CONFIG['deep_bucket_emb_cols']]
        test = pd.read_csv('toy_data/adult.test', header=None, names=CENSUS_COLUMNS)[WDL_CONFIG['deep_bucket_emb_cols']]
        total = pd.concat([train, test], axis=0)
        numeric_range = {}
        for col in WDL_CONFIG['deep_bucket_emb_cols']:
            numeric_range[col] = (total[col].min(), total[col].max())
        return numeric_range

    def _build_census_deep_columns(emb_dim=8, numeric_range=None):
        feature_columns = []
        for col in WDL_CONFIG['deep_emb_cols']:
            feature_columns.append(
                fc.embedding_column(fc.categorical_column_with_hash_bucket(col, hash_bucket_size=1000), dimension=emb_dim)
            )
        for col in WDL_CONFIG['deep_bucket_emb_cols']:
            feature_columns.append(
                fc.embedding_column(fc.bucketized_column(fc.numeric_column(col), boundaries=list(np.linspace(numeric_range[col][0], numeric_range[col][1], 1000))), dimension=emb_dim)
            )
        feat_field_size = len(feature_columns)
        return feature_columns, feat_field_size

    def _build_census_wide_columns(numeric_range=None):
        base_columns, cross_columns = [], []
        for col in WDL_CONFIG['wide_muti_hot_cols']:
            base_columns.append(
                fc.indicator_column(fc.categorical_column_with_hash_bucket(col, hash_bucket_size=1000))
            )
        for col in WDL_CONFIG['wide_bucket_cols']:
            base_columns.append(
                fc.bucketized_column(fc.numeric_column(col), boundaries=list(np.linspace(numeric_range[col][0], numeric_range['age'][1], 1000)))
            )
        for col in WDL_CONFIG['wide_cross_cols']:
            cross_columns.append(
                fc.indicator_column(fc.crossed_column([col[0], col[1]], hash_bucket_size=1000))
            )
        feature_columns = base_columns + cross_columns
        feat_field_size = len(feature_columns)
        return feature_columns,feat_field_size

    numeric_range = _get_numeric_feat_range()
    deep_columns, deep_fields_size = _build_census_deep_columns(emb_dim, numeric_range)
    wide_columns, wide_fields_size = _build_census_wide_columns(numeric_range)

    feat_config = {
        'deep_columns':     deep_columns,
        'deep_fields_size': deep_fields_size,
        'wide_columns':     wide_columns,
        'wide_fields_size': wide_fields_size,
        'embedding_dim':    emb_dim
    }
    return feat_config


def official_census_feature_columns_config_demo():
    # categorical_column
    gender = fc.categorical_column_with_vocabulary_list('gender', ['Female', 'Male'])
    education = fc.categorical_column_with_vocabulary_list('education', ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
    marital_status = fc.categorical_column_with_vocabulary_list('marital_status', ['Married-civ-spouse', 'Divorced', 'Married-spouse-absent','Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])
    relationship = fc.categorical_column_with_vocabulary_list('relationship', ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])
    workclass = fc.categorical_column_with_vocabulary_list('workclass', ['Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov', 'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    # To show an example of hashing:
    native_country = fc.categorical_column_with_hash_bucket('native_country', hash_bucket_size=1000)
    occupation = fc.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000)

    # Continuous feature columns.
    age = fc.numeric_column('age')
    education_num = fc.numeric_column('education_num')
    capital_gain = fc.numeric_column('capital_gain')
    capital_loss = fc.numeric_column('capital_loss')
    hours_per_week = fc.numeric_column('hours_per_week')

    # bucketized transformations.
    age_buckets = fc.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Wide columns and deep columns.
    base_columns = [gender, education, marital_status, relationship, workclass, occupation, native_country, age_buckets]
    crossed_columns = [
        fc.crossed_column(['education', 'occupation'], hash_bucket_size=1000),
        fc.crossed_column([age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
        fc.crossed_column(['native_country', 'occupation'], hash_bucket_size=1000)
    ]
    feature_columns = [
        fc.indicator_column(workclass),
        fc.indicator_column(education),
        fc.indicator_column(gender),
        fc.indicator_column(relationship),
        fc.embedding_column(native_country, dimension=32),
        fc.embedding_column(occupation, dimension=32),
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
    ]
    return feature_columns, base_columns, crossed_columns

