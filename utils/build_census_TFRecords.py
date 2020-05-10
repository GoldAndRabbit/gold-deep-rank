import os
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

CENSUS_COLUMNS = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation',
                  'relationship','race','gender','capital_gain','capital_loss','hours_per_week','native_country',
                  'income_bracket']

def serialize_census_example(age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week,
                             gender, education, marital_status, relationship, race, workclass,
                             native_country, occupation, income_bracket):

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

    def _int_list_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    feature = {
        # int feature
        'age'  :            _int64_feature(age),
        'fnlwgt':           _int64_feature(fnlwgt),
        'education_num':    _int64_feature(education_num),
        'capital_gain':     _int64_feature(capital_gain),
        'capital_loss':     _int64_feature(capital_loss),
        'hours_per_week':   _int64_feature(hours_per_week),
        # string feature
        'gender' :          _bytes_feature(gender),
        'education' :       _bytes_feature(education),
        'marital_status' :  _bytes_feature(marital_status),
        'relationship' :    _bytes_feature(relationship),
        'race':             _bytes_feature(race),
        'workclass' :       _bytes_feature(workclass),
        'native_country' :  _bytes_feature(native_country),
        'occupation' :      _bytes_feature(occupation),
        'income_bracket':   _bytes_feature(income_bracket)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def build_census_TFRecords(csv_file_dir, tfrecords_dir):
    def _load_data_from_csv(csv_file_dir):
        df_data = pd.read_csv(csv_file_dir, names=CENSUS_COLUMNS)
        df_data = df_data[[
            # int
            'age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week',
            # string
            'gender', 'education', 'marital_status', 'relationship', 'race',
            'workclass', 'native_country', 'occupation', 'income_bracket']]
        return df_data

    df_data = _load_data_from_csv(csv_file_dir)
    print('start writing census TFRecords...')
    with tf.io.TFRecordWriter(tfrecords_dir) as writer:
        for _, row in tqdm(df_data.iterrows()):
            example_str = serialize_census_example(
                # int
                row['age'], row['fnlwgt'], row['education_num'], row['capital_gain'], row['capital_loss'], row['hours_per_week'],
                # string
                row['gender'], row['education'], row['marital_status'], row['relationship'], row['race'],
                row['workclass'], row['native_country'], row['occupation'], row['income_bracket'])
            writer.write(example_str)
    print('finished writing census TFRecords...')


def parse_census_TFRecords_fn(record):
    features = {
        # int
        'age' :             tf.io.FixedLenFeature([], tf.int64),
        'fnlwgt' :          tf.io.FixedLenFeature([], tf.int64),
        'education_num' :   tf.io.FixedLenFeature([], tf.int64),
        'capital_gain' :    tf.io.FixedLenFeature([], tf.int64),
        'capital_loss' :    tf.io.FixedLenFeature([], tf.int64),
        'hours_per_week' :  tf.io.FixedLenFeature([], tf.int64),
        # string
        'gender':           tf.io.FixedLenFeature([], tf.string),
        'education':        tf.io.FixedLenFeature([], tf.string),
        'marital_status':   tf.io.FixedLenFeature([], tf.string),
        'relationship':     tf.io.FixedLenFeature([], tf.string),
        'race':             tf.io.FixedLenFeature([], tf.string),
        'workclass':        tf.io.FixedLenFeature([], tf.string),
        'native_country':   tf.io.FixedLenFeature([], tf.string),
        'occupation':       tf.io.FixedLenFeature([], tf.string),
        'income_bracket':   tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(record, features)
    return parsed


if __name__ == '__main__':
    train_csv_dir = os.getcwd().replace('utils', '/toy_data/adult.data')
    train_tfrecords_dir = os.getcwd().replace('utils', '/toy_data/census_adult.tfrecords')
    test_csv_dir = os.getcwd().replace('utils', '/toy_data/adult.test')
    test_tfrecords_dir = os.getcwd().replace('utils', '/toy_data/census_test.tfrecords')

    # build_census_TFRecords(csv_file_dir=train_csv_dir, tfrecords_dir=train_tfrecords_dir)
    # build_census_TFRecords(csv_file_dir=test_csv_dir, tfrecords_dir=test_tfrecords_dir)

    dataset = tf.data.TFRecordDataset(train_tfrecords_dir).map(parse_census_TFRecords_fn, num_parallel_calls=10).prefetch(500000)
    print(dataset)
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    with tf.Session() as sess:
        for i in range(2):
            print(sess.run(features))

