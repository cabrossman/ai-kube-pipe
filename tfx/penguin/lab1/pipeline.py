import tensorflow as tf
from tfx import v1 as tfx
import os
import urllib.request
import pprint
from absl import logging
import time
import glob
logging.set_verbosity(logging.INFO)  # Set default logging level.
pp = pprint.PrettyPrinter()



os.chdir('/Users/christopher.brossman/vs/tuts/kubeflow/ai-kube-pipe/tfx/penguin')
logging.info('changed directory to {}'.format(os.getcwd()))
NOW = time.strftime("%Y%m%d_%H%M%S")
ARTIFACT_STORE = os.path.join(os.sep, os.getcwd(),'artifact-store')
PIPELINE_NAME = 'penguin-simple'
PIPELINE_ROOT = os.path.join(ARTIFACT_STORE, NOW, 'pipelines', PIPELINE_NAME)
METADATA_PATH_DIR = os.path.join(ARTIFACT_STORE, NOW, 'metadata', PIPELINE_NAME)
METADATA_PATH = os.path.join(METADATA_PATH_DIR, 'metadata.db')
SERVING_MODEL_DIR = os.path.join(ARTIFACT_STORE, NOW,'serving_model', PIPELINE_NAME)
DATA_ROOT = os.path.join(ARTIFACT_STORE, 'data')
DATA = os.path.join(DATA_ROOT, "data.csv")

# DEUB
"""
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
context = InteractiveContext()

pipeline_name=PIPELINE_NAME
pipeline_root=PIPELINE_ROOT
data_root=DATA_ROOT
module_file='lab1/model/penguin_trainer.py'
serving_model_dir=SERVING_MODEL_DIR
metadata_path=METADATA_PATH
"""


#make dirs
for dirr in [PIPELINE_ROOT, METADATA_PATH_DIR, SERVING_MODEL_DIR, DATA_ROOT]:
    logging.info('Making directories if they dont exist : {}'.format(dirr))
    os.makedirs(dirr, exist_ok=True)

#download data
if not os.path.exists(os.path.join(DATA_ROOT,'data.csv')):
    logging.info('Downloading Data')
    _data_url = 'https://raw.githubusercontent.com/tensorflow/tfx/master/tfx/examples/penguin/data/labelled/penguins_processed.csv'
    urllib.request.urlretrieve(_data_url, DATA)


def create_pipeline(pipeline_name: str, 
                    pipeline_root: str, 
                    data_root: str,
                    module_file: str, 
                    serving_model_dir: str,
                    metadata_path: str) -> tfx.dsl.Pipeline:
  """Creates a three component penguin pipeline with TFX."""
  # Brings data into the pipeline.
  example_gen = tfx.components.CsvExampleGen(input_base=data_root)
  """
  from tensorflow import keras
  from tensorflow_transform.tf_metadata import schema_utils

  from tfx import v1 as tfx    
  from tfx_bsl.public import tfxio
  from tensorflow_metadata.proto.v0 import schema_pb2

  _FEATURE_KEYS = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
  _LABEL_KEY = 'species'

  _TRAIN_BATCH_SIZE = 20
  _EVAL_BATCH_SIZE = 10
  _FEATURE_SPEC = {
    **{
        feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)
           for feature in _FEATURE_KEYS
       },
    _LABEL_KEY: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
  }

  def _parse_function(example_proto):
    parsed_example = tf.io.parse_single_example(example_proto, _FEATURE_SPEC)
    label = parsed_example[_LABEL_KEY]
    features = {feature : parsed_example[feature] for feature in _FEATURE_KEYS}
    return (features, label)

  # Create a `TFRecordDataset` to read these files
  train_files=['/Users/christopher.brossman/vs/tuts/kubeflow/ai-kube-pipe/tfx/penguin/artifact-store/20210717_171343/pipelines/penguin-simple/CsvExampleGen/examples/1/Split-train/*']
  tfrecord_filenames = [f for train_file in train_files for f in glob.glob(train_file)]
  dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")
  dataset = dataset.map(_parse_function).batch(_TRAIN_BATCH_SIZE)
  dataset = dataset.repeat()

  model = _build_keras_model()
  model.fit(dataset, steps_per_epoch=1000)

  # Iterate over the first 3 records and decode them.
  for tfrecord in dataset.take(3):
    serialized_example = tfrecord.numpy()
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    pp.pprint(example)

  """
  # Uses user-provided Python function that trains a model.
  trainer = tfx.components.Trainer(
    module_file=module_file,
    examples=example_gen.outputs['examples'],
    train_args=tfx.proto.TrainArgs(num_steps=100),
    eval_args=tfx.proto.EvalArgs(num_steps=5)
  )

  # Pushes the model to a filesystem destination.
  push_destination = tfx.proto.PushDestination(
    filesystem=tfx.proto.PushDestination.Filesystem(base_directory=serving_model_dir)
  )
  pusher = tfx.components.Pusher(model=trainer.outputs['model'], push_destination=push_destination)

  # Following three components will be included in the pipeline.
  components = [example_gen, trainer, pusher]
  metadata_connection_config = tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path)

  pipeline = tfx.dsl.Pipeline(
    pipeline_name=pipeline_name,
    pipeline_root=pipeline_root,
    metadata_connection_config=metadata_connection_config,
    components=components
  )

  return pipeline

if __name__ == "__main__":

    pipeline = create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_root=DATA_ROOT,
            module_file='lab1/model/penguin_trainer.py',
            serving_model_dir=SERVING_MODEL_DIR,
            metadata_path=METADATA_PATH)
    tfx.orchestration.LocalDagRunner().run(pipeline)