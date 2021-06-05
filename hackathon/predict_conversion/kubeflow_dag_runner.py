from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import logging
from pipeline import configs
from pipeline import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.proto import trainer_pb2



OUTPUT_DIR = os.path.join('gs://', configs.GCS_BUCKET_NAME)
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, 'tfx_pipeline_output', configs.PIPELINE_NAME)
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, 'serving_model')
DATA_PATH = 'data'


def run():
  """Define a kubeflow pipeline."""

  metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
  tfx_image = os.environ.get('KUBEFLOW_TFX_IMAGE', None)

  runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
      kubeflow_metadata_config=metadata_config,
      tfx_image=tfx_image
  )

  # Set the SDK type label environment.
  os.environ[kubeflow_dag_runner.SDK_ENV_LABEL] = 'tfx-template'

  kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
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
          beam_pipeline_args=configs.DATAFLOW_BEAM_PIPELINE_ARGS,
          ai_platform_training_args=configs.GCP_AI_PLATFORM_TRAINING_ARGS,
          ai_platform_serving_args=configs.GCP_AI_PLATFORM_SERVING_ARGS,
      ))


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  run()
