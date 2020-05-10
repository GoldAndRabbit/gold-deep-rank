import shutil
import tensorflow as tf
from deep_ctr_models.din import din_model_fn


def build_estimator(model_dir, model_type, paras):
    run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 0}))
    if model_type == 'din':
        return tf.estimator.Estimator(model_fn=din_model_fn, model_dir=model_dir, config=run_config)
    else:
        print('error')


def input_fn_from_tfrecords(data_file, num_epochs, shuffle, batch_size):
    assert tf.io.gfile.exists(data_file), ('no file named : ' + str(data_file))

    def _parse_ama_ele_TFRecords_fn(record):
        features = {
            # int
            'user_id':   tf.io.FixedLenFeature([], tf.int64),
            'item_id':   tf.io.FixedLenFeature([], tf.int64),
            'item_cate': tf.io.FixedLenFeature([], tf.int64),
            'label':     tf.io.FixedLenFeature([], tf.int64),
            # int list
            # 'seq':       tf.io.FixedLenFeature([20], tf.int64),
            # 'seq_cate':  tf.io.FixedLenFeature([20], tf.int64),
        }
        parsed = tf.io.parse_single_example(record, features)
        labels = parsed.pop('label')
        labels = tf.reshape(labels, [-1])
        labels = tf.to_float(labels)
        return features, labels

    dataset = tf.data.TFRecordDataset(data_file).map(_parse_ama_ele_TFRecords_fn, num_parallel_calls=10)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def train_ama_ele_from_tfrecords():
    all_paras = {
        'model_dir': './seq_ckpt_dir/',
        'model_type': 'din',
        'train_epoches': 4,
        'epoches_per_eval': 2,
        'train_data_tfrecords_dir': 'toy_data/ama_ele_train_pad.tfrecords',
        'test_data_tfrecords_dir':  'toy_data/ama_ele_test_pad.tfrecords',
        'batch_size': 8,
    }
    print('using: ' +  all_paras['model_type'] + ' model...')
    shutil.rmtree(all_paras['model_dir'], ignore_errors=True)
    model = build_estimator(all_paras['model_dir'], all_paras['model_type'], paras=all_paras)
    model.train(
        input_fn=lambda: input_fn_from_tfrecords(
            data_file=all_paras['train_data_tfrecords_dir'],
            num_epochs=all_paras['epoches_per_eval'],
            shuffle=True,
            batch_size=all_paras['batch_size']
        )
    )
    results = model.evaluate(
        input_fn=lambda: input_fn_from_tfrecords(
            data_file=all_paras['test_data_tfrecords_dir'],
            num_epochs=1,
            shuffle=False,
            batch_size=all_paras['batch_size']
        )
    )
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.set_random_seed(1)
    train_ama_ele_from_tfrecords()

