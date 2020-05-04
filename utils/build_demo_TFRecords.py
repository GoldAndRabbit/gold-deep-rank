from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf


def serialize_example(age, name, score, label):

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        # string encode to bytes
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

    # define feature type
    feature = {
        'age'  : _int64_feature(age),
        'name' : _bytes_feature(name),
        'score': _float_feature(score),
        'label': _int64_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def serialize_demo():
    serialized_example = serialize_example(4, 'goat', 0.9876, 1)
    example_proto = tf.train.Example.FromString(serialized_example)
    print(example_proto)


def mock_test_data():
    n_observations = int(1e4)
    age = np.random.randint(0, 5, size=n_observations)             # int feature
    strings = np.array(['cat', 'dog', 'chicken', 'horse', 'goat'])
    name = strings[age]                                            # byte(string)feature
    score = np.random.randn(n_observations)                        # float feature
    label = np.random.randint(0, 2, size=n_observations)
    df_data = pd.DataFrame()
    df_data['age'] = age
    df_data['name'] = name
    df_data['score'] = score
    df_data['label'] = label
    return df_data


def build_demo_tf_records(df_data,file_dir):
    print('start writing TFRecords...')
    with tf.io.TFRecordWriter(file_dir) as writer:
        for _, row in tqdm(df_data.iterrows()):
            example_str = serialize_example(row['age'], row['name'], row['score'], row['label'])
            writer.write(example_str)
    print('finished writing TFRecords...')


def parse_tf_records_fn(record):
    features = {
        'age'  : tf.io.FixedLenFeature([], tf.int64),
        'name' : tf.io.FixedLenFeature([], tf.string),
        'score': tf.io.FixedLenFeature([], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(record, features)
    return parsed

if __name__ == '__main__':
    file_path = 'toy_data/demo.tfrecords'
    file_paths = [file_path]
    build_demo_tf_records(mock_test_data(),file_path)
    dataset = tf.data.TFRecordDataset(file_path).map(parse_tf_records_fn, num_parallel_calls=10).prefetch(500000)
    print(dataset)
