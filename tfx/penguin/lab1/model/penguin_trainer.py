from typing import List
from absl import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow_transform.tf_metadata import schema_utils

from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2

_FEATURE_KEYS = [
    'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g'
]
_LABEL_KEY = 'species'

_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10

# Since we're not generating or creating a schema, we will instead create
# a feature spec.  Since there are a fairly small number of features this is
# manageable for this dataset.
_FEATURE_SPEC = {
    **{
        feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)
           for feature in _FEATURE_KEYS
       },
    _LABEL_KEY: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
}


def _input_fn(file_pattern: List[str],
              data_accessor: tfx.components.DataAccessor,
              schema: schema_pb2.Schema,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    schema: schema of the input data.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_LABEL_KEY),
      schema=schema).repeat()


def _build_keras_model() -> tf.keras.Model:
  """Creates a DNN Keras model for classifying penguin data.

  Returns:
    A Keras Model.
  """
  # The model below is built with Functional API, please refer to
  # https://www.tensorflow.org/guide/keras/overview for all API options.
  inputs = [keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]
  d = keras.layers.concatenate(inputs)
  for _ in range(2):
    d = keras.layers.Dense(8, activation='relu')(d)
  outputs = keras.layers.Dense(3)(d)

  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
      optimizer=keras.optimizers.Adam(1e-2),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[keras.metrics.SparseCategoricalAccuracy()])

  model.summary(print_fn=logging.info)
  return model


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
  """Train the model based on given args.
  https://www.tensorflow.org/tfx/api_docs/python/tfx/components/trainer/fn_args_utils/FnArgs
  EXAMPLES: 
  FnArgs(
        working_dir=None, 
        train_files=['/Users/christopher.brossman/vs/tuts/kubeflow/ai-kube-pipe/tfx/penguin/artifact-store/20210717_171343/pipelines/penguin-simple/CsvExampleGen/examples/1/Split-train/*'], 
        eval_files=['/Users/christopher.brossman/vs/tuts/kubeflow/ai-kube-pipe/tfx/penguin/artifact-store/20210717_171343/pipelines/penguin-simple/CsvExampleGen/examples/1/Split-eval/*'], 
        train_steps=100, 
        eval_steps=5, 
        schema_path=None, 
        schema_file=None, 
        transform_graph_path=None, 
        transform_output=None, 
        data_accessor=DataAccessor(tf_dataset_factory=<function get_tf_dataset_factory_from_artifact.<locals>.dataset_factory at 0x7fcb7a579200>, 
            record_batch_factory=<function get_record_batch_factory_from_artifact.<locals>.record_batch_factory at 0x7fcb99f000e0>, 
            data_view_decode_fn=None), 
        serving_model_dir='/Users/christopher.brossman/vs/tuts/kubeflow/ai-kube-pipe/tfx/penguin/artifact-store/20210717_171343/pipelines/penguin-simple/Trainer/model/2/Format-Serving', 
        eval_model_dir='/Users/christopher.brossman/vs/tuts/kubeflow/ai-kube-pipe/tfx/penguin/artifact-store/20210717_171343/pipelines/penguin-simple/Trainer/model/2/Format-TFMA', 
        model_run_dir='/Users/christopher.brossman/vs/tuts/kubeflow/ai-kube-pipe/tfx/penguin/artifact-store/20210717_171343/pipelines/penguin-simple/Trainer/model_run/2', 
        base_model=None, 
        hyperparameters=None, 
        custom_config=None
    )

    https://www.tensorflow.org/tutorials/load_data/tfrecord
    https://github.com/tensorflow/tfx/blob/v0.30.1/tfx/components/trainer/fn_args_utils.py

    from tfx.components.util import tfxio_utils
    data_accessor = tfxio_utils.get_tf_dataset_factory_from_artifact(example_gen.outputs['examples'], ['Trainer'])

    ({'body_mass_g': <tf.Tensor: shape=(20, 1), dtype=float32, numpy=
array([[0.5416667 ],
       [0.4375    ],
       [0.3263889 ],
       [0.41666666],
       [0.44444445],
       [0.5694444 ],
       [0.33333334],
       [0.5972222 ],
       [0.7916667 ],
       [0.6388889 ],
       [0.125     ],
       [0.41666666],
       [0.43055555],
       [0.30555555],
       [0.2777778 ],
       [0.05555556],
       [0.5972222 ],
       [0.8333333 ],
       [0.4861111 ],
       [0.05555556]], dtype=float32)>, 'culmen_depth_mm': <tf.Tensor: shape=(20, 1), dtype=float32, numpy=
array([[0.07142857],
       [0.7619048 ],
       [0.64285713],
       [0.08333334],
       [0.6904762 ],
       [0.08333334],
       [0.8214286 ],
       [0.22619048],
       [0.35714287],
       [0.29761904],
       [0.45238096],
       [0.17857143],
       [0.5714286 ],
       [0.6904762 ],
       [0.47619048],
       [0.5952381 ],
       [0.1904762 ],
       [0.3452381 ],
       [0.11904762],
       [0.35714287]], dtype=float32)>, 'culmen_length_mm': <tf.Tensor: shape=(20, 1), dtype=float32, numpy=
array([[0.48727274],
       [0.36727273],
       [0.33818182],
       [0.48      ],
       [0.31636363],
       [0.47636363],
       [0.22181818],
       [0.59636366],
       [0.61454546],
       [0.52      ],
       [0.13090909],
       [0.49818182],
       [0.27636364],
       [0.11636364],
       [0.14181818],
       [0.08727273],
       [0.45090908],
       [0.6363636 ],
       [0.6036364 ],
       [0.03636364]], dtype=float32)>, 'flipper_length_mm': <tf.Tensor: shape=(20, 1), dtype=float32, numpy=
array([[0.7118644 ],
       [0.42372882],
       [0.5084746 ],
       [0.6101695 ],
       [0.6101695 ],
       [0.7288136 ],
       [0.30508474],
       [0.7966102 ],
       [0.7457627 ],
       [0.8305085 ],
       [0.22033899],
       [0.6440678 ],
       [0.3559322 ],
       [0.2542373 ],
       [0.2542373 ],
       [0.2542373 ],
       [0.7118644 ],
       [0.89830506],
       [0.6440678 ],
       [0.10169491]], dtype=float32)>}, <tf.Tensor: shape=(20, 1), dtype=int64, numpy=
array([[2],
       [0],
       [0],
       [2],
       [0],
       [2],
       [0],
       [2],
       [2],
       [2],
       [0],
       [2],
       [0],
       [0],
       [0],
       [0],
       [2],
       [2],
       [2],
       [0]])>)

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """

  # This schema is usually either an output of SchemaGen or a manually-curated
  # version provided by pipeline author. A schema can also derived from TFT
  # graph if a Transform component is used. In the case when either is missing,
  # `schema_from_feature_spec` could be used to generate schema from very simple
  # feature_spec, but the schema returned would be very primitive.
  schema = schema_utils.schema_from_feature_spec(_FEATURE_SPEC)

  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      schema,
      batch_size=_TRAIN_BATCH_SIZE)
  eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      schema,
      batch_size=_EVAL_BATCH_SIZE)
  print('LOOK FOR ME!')
  print('type {}'.format(type(train_dataset)))
  print('train_dataset {}'.format(train_dataset))
  for tfrecord in train_dataset.take(3):
      print(tfrecord)

  model = _build_keras_model()
  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  # The result of the training should be saved in `fn_args.serving_model_dir`
  # directory.
  model.save(fn_args.serving_model_dir, save_format='tf')