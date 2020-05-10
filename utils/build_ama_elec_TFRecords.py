import os

import pandas as pd
import tensorflow as tf
from tqdm import tqdm


# def serialize_ama_ele_example(user_id, item_id, item_cate, label, seq, seq_cate):
def serialize_ama_ele_example(user_id,item_id,item_cate,label):

    def _int64_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _float_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

    feature = {
        # int feature
        'user_id':   _int64_feature(user_id),
        'item_id':   _int64_feature(item_id),
        'item_cate': _int64_feature(item_cate),
        'label':     _int64_feature(label),
        # int list feature
        # 'seq':       _bytes_feature(seq),
        # 'seq_cate':  _bytes_feature(seq_cate),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def build_ama_ele_TFRecords(csv_file_dir, tfrecords_dir):
    def _load_data_from_csv(csv_file_dir):
        df_data = pd.read_csv(csv_file_dir)
        df_data = df_data[[
            # int
            'user_id', 'item_id', 'item_cate', 'label',
            # int list
            'seq', 'seq_cate']]
        return df_data

    df_data = _load_data_from_csv(csv_file_dir)
    df_data = df_data.head(100)
    print('start writing ama elec TFRecords...')
    with tf.io.TFRecordWriter(tfrecords_dir) as writer:
        for _, row in tqdm(df_data.iterrows()):
            example_str = serialize_ama_ele_example(
                # int
                row['user_id'], row['item_id'], row['item_cate'], row['label']
                # int list
                # [int(x) for x in row['seq'].split(',')],
                # [int(x) for x in row['seq_cate'].split(',')]
                # [row['seq'].split(',')],
                # [row['seq_cate'].split(',')],
                # [str(x).encode() for x in row['seq'].split(',')],
                # [str(x).encode() for x in row['seq_cate'].split(',')],
            )
            writer.write(example_str)
    print('finished writing ama elec TFRecords...')


def parse_ama_ele_TFRecords_fn(record):
    features = {
        # int
        'user_id':   tf.io.FixedLenFeature([], tf.int64),
        'item_id':   tf.io.FixedLenFeature([], tf.int64),
        'item_cate': tf.io.FixedLenFeature([], tf.int64),
        'label':     tf.io.FixedLenFeature([], tf.int64),
        # int list feature
        # 'seq':       tf.io.FixedLenFeature([20], tf.int64),
        # 'seq_cate':  tf.io.FixedLenFeature([20], tf.int64),
    }
    parsed = tf.io.parse_single_example(record, features)
    return parsed


if __name__ == '__main__':
    train_csv_dir = os.getcwd().replace('utils', '/toy_data/ama_ele_train_pad.csv')
    train_tfrecords_dir = os.getcwd().replace('utils', '/toy_data/ama_ele_train_pad.tfrecords')
    test_csv_dir = os.getcwd().replace('utils', '/toy_data/ama_ele_test_pad.csv')
    test_tfrecords_dir = os.getcwd().replace('utils', '/toy_data/ama_ele_test_pad.tfrecords')

    build_ama_ele_TFRecords(csv_file_dir=train_csv_dir, tfrecords_dir=train_tfrecords_dir)
    build_ama_ele_TFRecords(csv_file_dir=test_csv_dir, tfrecords_dir=test_tfrecords_dir)

    dataset = tf.data.TFRecordDataset(train_tfrecords_dir).map(parse_ama_ele_TFRecords_fn, num_parallel_calls=10).prefetch(500000)
    print(dataset)
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    with tf.Session() as sess:
        for i in range(2):
            print(sess.run(features))
