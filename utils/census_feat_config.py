import tensorflow.feature_column as fc
import pandas as pd
import numpy as np

CENSUS_COLUMNS = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','gender','capital_gain','capital_loss','hours_per_week','native_country','income_bracket']
CENSUS_COLUMN_DEFAULTS = [[-1],[''],[-1],[''],[-1],[''],[''],[''],[''],[''],[-1],[-1],[-1],[''],['']]
SEQ_COLUMNS = ['user_id', 'seq', 'item_id', 'seq_cate', 'item_cate', 'label']
SEQ_COLUMNS_DEFAULTS = [[''], [''], [''], [''], [''], ['0']]


def build_ama_ele_columns():
    feature_columns = [
        fc.embedding_column(fc.categorical_column_with_hash_bucket('user_id',   hash_bucket_size=200000), dimension=32),
        fc.embedding_column(fc.categorical_column_with_hash_bucket('item_id',   hash_bucket_size=70000),  dimension=32),
        fc.embedding_column(fc.categorical_column_with_hash_bucket('item_cate', hash_bucket_size=1000),   dimension=32),
    ]
    feat_field_size = len(feature_columns)
    return feature_columns, feat_field_size


def get_census_numeric_feat_range():
    train = pd.read_csv('./toy_data/adult.data',header=None,names=CENSUS_COLUMNS)[['age','education_num','capital_gain','capital_loss','hours_per_week']]
    test = pd.read_csv('./toy_data/adult.test',header=None,names=CENSUS_COLUMNS)[['age','education_num','capital_gain','capital_loss','hours_per_week']]
    total = pd.concat([train, test], axis=0)
    n_range = dict()
    n_range['age'] = (total['age'].min(), total['age'].max())
    n_range['education_num'] = (total['education_num'].min(), total['education_num'].max())
    n_range['capital_gain'] = (total['capital_gain'].min(), total['capital_gain'].max())
    n_range['capital_loss'] = (total['capital_loss'].min(), total['capital_loss'].max())
    n_range['hours_per_week'] = (total['hours_per_week'].min(), total['hours_per_week'].max())
    return n_range


def build_census_wide_columns():
    n_range = get_census_numeric_feat_range()
    base_columns = [
        fc.bucketized_column(fc.numeric_column('age'), boundaries=list(np.linspace(n_range['age'][0], n_range['age'][1], 1000))),
        fc.bucketized_column(fc.numeric_column('education_num'), boundaries=list(np.linspace(n_range['education_num'][0], n_range['education_num'][1], 1000))),
        fc.bucketized_column(fc.numeric_column('capital_gain'), boundaries=list(np.linspace(n_range['capital_gain'][0], n_range['capital_gain'][1], 1000))),
        fc.bucketized_column(fc.numeric_column('capital_loss'), boundaries=list(np.linspace(n_range['capital_loss'][0], n_range['capital_loss'][1], 1000))),
        fc.bucketized_column(fc.numeric_column('hours_per_week'), boundaries=list(np.linspace(n_range['hours_per_week'][0], n_range['hours_per_week'][1], 1000))),
        fc.indicator_column(fc.categorical_column_with_hash_bucket('gender', hash_bucket_size=1000)),
        fc.indicator_column(fc.categorical_column_with_hash_bucket('education', hash_bucket_size=1000)),
        fc.indicator_column(fc.categorical_column_with_hash_bucket('marital_status', hash_bucket_size=1000)),
        fc.indicator_column(fc.categorical_column_with_hash_bucket('relationship', hash_bucket_size=1000)),
        fc.indicator_column(fc.categorical_column_with_hash_bucket('workclass', hash_bucket_size=1000)),
        fc.indicator_column(fc.categorical_column_with_hash_bucket('native_country', hash_bucket_size=1000)),
        fc.indicator_column(fc.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000))
    ]
    age_buckets = fc.bucketized_column(fc.numeric_column("age"), boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    cross_columns = [
        fc.indicator_column(fc.crossed_column(["education", "occupation"], hash_bucket_size=1000)),
        fc.indicator_column(fc.crossed_column(["native_country", "occupation"], hash_bucket_size=1000)),
        fc.indicator_column(fc.crossed_column([age_buckets, "education", "occupation"], hash_bucket_size=1000))
    ]
    feature_columns = base_columns + cross_columns
    feat_field_size = len(feature_columns)
    return feature_columns, feat_field_size


def build_census_emb_columns():
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
        fc.embedding_column(fc.categorical_column_with_hash_bucket('marital_status', hash_bucket_size=1000), dimension=32),
        fc.embedding_column(fc.categorical_column_with_hash_bucket('relationship', hash_bucket_size=1000), dimension=32),
        fc.embedding_column(fc.categorical_column_with_hash_bucket('workclass', hash_bucket_size=1000), dimension=32),
        fc.embedding_column(fc.categorical_column_with_hash_bucket('native_country', hash_bucket_size=1000), dimension=32),
        fc.embedding_column(fc.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000), dimension=32)
    ]
    feat_field_size = len(feature_columns)
    return feature_columns, feat_field_size


def official_feature_columns_config_demo():
    # categorical_column
    gender = fc.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])
    education = fc.categorical_column_with_vocabulary_list("education", ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th", "Preschool", "12th"])
    marital_status = fc.categorical_column_with_vocabulary_list("marital_status", ["Married-civ-spouse", "Divorced", "Married-spouse-absent","Never-married", "Separated", "Married-AF-spouse", "Widowed"])
    relationship = fc.categorical_column_with_vocabulary_list("relationship", ["Husband", "Not-in-family", "Wife", "Own-child", "Unmarried", "Other-relative"])
    workclass = fc.categorical_column_with_vocabulary_list("workclass", ["Self-emp-not-inc", "Private", "State-gov", "Federal-gov", "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"])

    # To show an example of hashing:
    native_country = fc.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)
    occupation = fc.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)

    # Continuous feature columns.
    age = fc.numeric_column("age")
    education_num = fc.numeric_column("education_num")
    capital_gain = fc.numeric_column("capital_gain")
    capital_loss = fc.numeric_column("capital_loss")
    hours_per_week = fc.numeric_column("hours_per_week")

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

