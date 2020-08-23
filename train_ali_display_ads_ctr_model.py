import os
import shutil
import tensorflow as tf
from feat_config.ali_display_ads_feat_config import ALI_DISPLAY_ADS_CONFIG, build_ali_display_ads_feat_columns
from deep_ctr_models.wdl     import wdl_estimator
from deep_ctr_models.dcn     import dcn_model_fn
from deep_ctr_models.autoint import autoint_model_fn
from deep_ctr_models.xdeepfm import xdeepfm_model_fn
from deep_ctr_models.deepfm  import deepfm_model_fn
from deep_ctr_models.resnet  import res_model_fn
from deep_ctr_models.fibinet import fibinet_model_fn
from deep_ctr_models.afm     import afm_model_fn
from deep_ctr_models.pnn     import pnn_model_fn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def ali_display_ads_input_fn_from_csv_file(data_file, num_epochs, shuffle, batch_size):
    assert tf.io.gfile.exists(data_file), ('no file named : ' + str(data_file))

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=ALI_DISPLAY_ADS_CONFIG['columns_defaults'])
        features = dict(zip(ALI_DISPLAY_ADS_CONFIG['columns'], columns))
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


def build_estimator(ckpt_dir, model_name, params_config):
    model_fn_map = params_config['model_fn_map']
    assert model_name in model_fn_map.keys(), ('no model named : ' + str(model_name))
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}),
        save_checkpoints_steps=2000,
        save_summary_steps=500,
        log_step_count_steps=500,
        keep_checkpoint_max=3
    )
    if model_name is 'wdl':
        return wdl_estimator(params=params_config, config=run_config)
    else:
        return tf.estimator.Estimator(model_fn=model_fn_map[model_name], model_dir=ckpt_dir, config=run_config, params=params_config)


def train_ali_display_ads_data():
    feat_columns = build_ali_display_ads_feat_columns(emb_dim=8)
    MODEL_FN_MAP = {
        'wdl':      wdl_estimator,
        'dcn':      dcn_model_fn,
        'autoint':  autoint_model_fn,
        'xdeepfm':  xdeepfm_model_fn,
        'deepfm':   deepfm_model_fn,
        'resnet':   res_model_fn,
        'pnn':      pnn_model_fn,
        'fibinet':  fibinet_model_fn,
        'afm':      afm_model_fn,
    }
    ARGS = {
        # data/ckpt dir config
        'train_data_dir':           ALI_DISPLAY_ADS_CONFIG['data_path'] + 'train_log_sample.csv',
        'test_data_dir':            ALI_DISPLAY_ADS_CONFIG['data_path'] + 'test_log_sample.csv',
        'load_tf_records_data':     False,
        'ckpt_dir':                 ALI_DISPLAY_ADS_CONFIG['data_path'] + 'ckpt_dir/',
        # traning process config
        'shuffle':                  True,
        'model_name':               'wdl',
        'optimizer':                'adam',
        'train_epoches_num':        1,
        'batch_size':               16,
        'epoches_per_eval':         2,
        'learning_rate':            0.01,
        'deep_layer_nerouns':       [256, 128, 64],
        'embedding_dim':            feat_columns['embedding_dim'],
        'deep_columns':             feat_columns['deep_columns'],
        'deep_fields_size':         feat_columns['deep_fields_size'],
        'wide_columns':             feat_columns['wide_columns'],
        'wide_fields_size':         feat_columns['wide_fields_size'],
        'model_fn_map':             MODEL_FN_MAP,
        'fibinet':                  {'pooling': 'max', 'reduction_ratio': 2}
    }
    print('this process will train a: ' + ARGS['model_name'] + ' model...')
    shutil.rmtree(ARGS['ckpt_dir'], ignore_errors=True)
    model = build_estimator(ARGS['ckpt_dir'], ARGS['model_name'], params_config=ARGS)
    model.train(
        input_fn=lambda: ali_display_ads_input_fn_from_csv_file(data_file=ARGS['train_data_dir'], num_epochs=ARGS['train_epoches_num'], shuffle=True, batch_size=ARGS['batch_size'])
    )
    results = model.evaluate(
        input_fn=lambda: ali_display_ads_input_fn_from_csv_file(data_file=ARGS['test_data_dir'], num_epochs=1, shuffle=False, batch_size=ARGS['batch_size']
        )
    )
    predictions = model.predict(
        input_fn=lambda: ali_display_ads_input_fn_from_csv_file(data_file=ARGS['test_data_dir'], num_epochs=1, shuffle=False, batch_size=ARGS['batch_size'])
    )
    # for x in predictions:
    #     print(x['probabilities'][0])
    #     print(x['label'][0])



if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.set_random_seed(1)
    train_ali_display_ads_data()
