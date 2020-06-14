import shutil
import os
import tensorflow as tf
from utils.ali_display_ads_feat_config import build_ali_display_ads_columns, build_ali_display_ads_wide_columns, ALI_DISPLAY_ADS_COLUMNS, ALI_DISPLAY_ADS_COLUMN_DEFAULTS
from deep_ctr_models.dcn import dcn_model_fn
from deep_ctr_models.autoint import autoint_model_fn
from deep_ctr_models.xdeepfm import xdeepfm_model_fn
from deep_ctr_models.deepfm import deepfm_model_fn
from deep_ctr_models.resnet import res_model_fn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def ali_display_ads_input_fn_from_csv_file(data_file, num_epochs, shuffle, batch_size):
    assert tf.io.gfile.exists(data_file), ('no file named : ' + str(data_file))

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=ALI_DISPLAY_ADS_COLUMN_DEFAULTS)
        features = dict(zip(ALI_DISPLAY_ADS_COLUMNS, columns))
        labels = tf.equal(features.pop('clk'), 1)
        labels = tf.reshape(labels, [-1])
        labels = tf.to_float(labels)
        return features, labels

    dataset = tf.data.TextLineDataset(data_file)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def build_estimator(model_name, params_config):
    if model_name == 'wdl':
        columns, feat_field_size = build_ali_display_ads_columns(emb_dims=params_config['embedding_dim'])
        run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 0}))
        return tf.estimator.DNNLinearCombinedClassifier(model_dir=params_config['ckpt_dir'],
                                                        linear_feature_columns=params_config['wide_feat_columns'],
                                                        # linear_feature_columns=None,
                                                        dnn_feature_columns=params_config['columns'],
                                                        dnn_hidden_units=params_config['deep_layer_nerouns'],
                                                        config=run_config)
    else:
        MODEL_FN_MAP = {
            'dcn':          dcn_model_fn,
            'autoint':      autoint_model_fn,
            'xdeepfm':      xdeepfm_model_fn,
            'deepfm':       deepfm_model_fn,
            'res_model_fn': res_model_fn,
        }
        assert model_name in MODEL_FN_MAP.keys(), ('no model named : ' + str(model_name))
        run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 0}))
        return tf.estimator.Estimator(model_fn=MODEL_FN_MAP[model_name],
                                      model_dir=params_config['ckpt_dir'],
                                      config=run_config,
                                      params=params_config)


def train_ali_display_ads_data():
    columns, feat_field_size = build_ali_display_ads_columns(emb_dims=8)
    wide_feat_columns, wide_field_size = build_ali_display_ads_wide_columns()
    PATH = '/media/psdz/hdd/Download/ali_display_ads/'
    params_config = {
        # 'train_data_dir': PATH + 'train_log_sample.csv',
        # 'test_data_dir': PATH + 'test_log_sample.csv',
        'train_data_dir': PATH + 'train_log.csv',
        'test_data_dir': PATH + 'test_log.csv',
        'ckpt_dir': PATH + 'ali_display_ads_ckpt_dir/',
        'model_name': 'wdl',
        'batch_size': 1024,
        'epoches_per_eval': 1,
        'learning_rate': 0.01,
        'optimizer': 'adam',
        'shuffle': False,
        'embedding_dim': 8,
        'deep_layer_nerouns': [512, 512, 512],
        'columns': columns,
        'feat_field_size': feat_field_size,
        'wide_feat_columns': wide_feat_columns,
        'wide_field_size': wide_field_size
    }
    print('this process will train a: ' + params_config['model_name'] + ' model...')
    shutil.rmtree(params_config['ckpt_dir'], ignore_errors=True)
    model = build_estimator(params_config['model_name'], params_config=params_config)
    model.train(
        input_fn=lambda: ali_display_ads_input_fn_from_csv_file(
            data_file=params_config['train_data_dir'],
            num_epochs=params_config['epoches_per_eval'],
            shuffle=True if params_config['shuffle']==True else False,
            batch_size=params_config['batch_size']
        )
    )

    results = model.evaluate(
        input_fn=lambda: ali_display_ads_input_fn_from_csv_file(
            data_file=params_config['test_data_dir'],
            num_epochs=1,
            shuffle=False,
            batch_size=params_config['batch_size']
        )
    )
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.set_random_seed(1)
    train_ali_display_ads_data()

