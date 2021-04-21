# import tensorflow as tf
import tensorflow.compat.v1 as tf


def wdl_estimator(params, config):
    deep_columns     = params['deep_columns']
    deep_fields_size = params['deep_fields_size']
    org_emb_size     = params['embedding_dim']
    wide_columns     = params['wide_columns']
    wide_fields_size = params['wide_fields_size']

    dnn_optimizer = tf.train.ProximalAdagradOptimizer(learning_rate= 0.01, l1_regularization_strength=0.001, l2_regularization_strength=0.001)

    wdl_estimator = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=params['ckpt_dir'],
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_optimizer=dnn_optimizer,
        dnn_dropout=0.1,
        batch_norm=False,
        dnn_hidden_units=params['deep_layer_nerouns'],
        config=config
    )

    return wdl_estimator