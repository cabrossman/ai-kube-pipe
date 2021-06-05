from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import logging

from pipeline import configs
from pipeline import pipeline
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import trainer_pb2

OUTPUT_DIR = '.'
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, 'tfx_pipeline_output',
                             configs.PIPELINE_NAME)
METADATA_PATH = os.path.join(OUTPUT_DIR, 'tfx_metadata', configs.PIPELINE_NAME,
                             'metadata.db')
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, 'serving_model')
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def run():
  """Define a beam pipeline."""

  BeamDagRunner().run(
      pipeline.create_pipeline(
          pipeline_name=configs.PIPELINE_NAME,
          pipeline_root=PIPELINE_ROOT,
          data_path=DATA_PATH,
          query=configs.BIG_QUERY_QUERY,
          preprocessing_fn=configs.PREPROCESSING_FN,
          run_fn=configs.RUN_FN,
          train_args=trainer_pb2.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS),
          eval_args=trainer_pb2.EvalArgs(num_steps=configs.EVAL_NUM_STEPS),
          eval_accuracy_threshold=configs.EVAL_ACCURACY_THRESHOLD,
          serving_model_dir=SERVING_MODEL_DIR,
          # beam_pipeline_args=configs.BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS,
          metadata_connection_config=metadata.sqlite_metadata_connection_config(
              METADATA_PATH)))


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  run()
