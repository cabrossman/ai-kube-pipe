import absl
import os
import tempfile
import time

import tensorflow as tf
import tensorflow_data_validation as tfdv
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
import tfx

from pprint import pprint
from tensorflow_metadata.proto.v0 import schema_pb2, statistics_pb2, anomalies_pb2
from tensorflow_transform.tf_metadata import schema_utils
from tfx.components import CsvExampleGen
from tfx.extensions.google_cloud_big_query.example_gen.component import BigQueryExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import InfraValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components import Tuner
from tfx.dsl.components.base import executor_spec
from tfx.components.common_nodes.importer_node import ImporterNode
from tfx.components.trainer import executor as trainer_executor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx import proto
from tfx.proto import evaluator_pb2
from tfx.proto import example_gen_pb2
from tfx.proto import infra_validator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.proto.evaluator_pb2 import SingleSlicingSpec

from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import HyperParameters
from tfx.types.standard_artifacts import ModelBlessing
from tfx.types.standard_artifacts import InfraBlessing

### Assume CWD is in ai-kube-pipe ex: '/Users/christopher.brossman/vs/tuts/kubeflow/ai-kube-pipe/tfx/tfx1'
os.chdir('/Users/christopher.brossman/vs/tuts/kubeflow/ai-kube-pipe/tfx/tfx1/lab')
from utils import make_schema, hypertune


NOW = time.strftime("%Y%m%d_%H%M%S")
ARTIFACT_STORE = os.path.join(os.sep, os.getcwd(),'artifact-store')
SERVING_MODEL_DIR = os.path.join(os.sep, os.getcwd(),'serving_model')
DATA_ROOT = 'gs://workshop-datasets/covertype/small'
PIPELINE_NAME = 'tfx-covertype-classifier'
PIPELINE_ROOT = os.path.join(ARTIFACT_STORE, PIPELINE_NAME, NOW)
os.makedirs(PIPELINE_ROOT, exist_ok=True)

GCLOUD_PROJECT = os.getenv('PROJECT_ID')

JOB_NAME = '{}-{}'.format(PIPELINE_NAME,NOW)
STORAGE_LOC = 'gs://test-tmp-storage/{}'.format(JOB_NAME)

BEAM_PIPELINE_ARGS = [
    '--project', GCLOUD_PROJECT, 
    '--job_name', JOB_NAME,
    '--temp_location', '{}/data'.format(STORAGE_LOC),
    '--staging_location', '{}/code'.format(STORAGE_LOC),
    '--region', 'us-central1'
    ]

SCHEMA_DIR = os.path.join(ARTIFACT_STORE, 'schema')
SCHEMA_FILE = os.path.join(SCHEMA_DIR, 'schema.pbtxt')
HYPERTUNE = False

### Start session interactively
context = InteractiveContext(
    pipeline_name=PIPELINE_NAME,
    pipeline_root=PIPELINE_ROOT,
    metadata_connection_config=None)


### Get Data
QUERY_TEMPLATE = 'SELECT * FROM trr-assignments-staging.covertype_dataset.{}'
input = example_gen_pb2.Input(splits=[
                example_gen_pb2.Input.Split(name='train', pattern=QUERY_TEMPLATE.format('training')),
                example_gen_pb2.Input.Split(name='eval', pattern=QUERY_TEMPLATE.format('validation'))
            ])
example_gen = BigQueryExampleGen(input_config=input)
context.run(example_gen, beam_pipeline_args=BEAM_PIPELINE_ARGS)

if not os.path.isdir(SCHEMA_DIR):
    make_schema(context, example_gen, SCHEMA_DIR, SCHEMA_FILE)


### Get the external Schema - ImporterNode
schema_importer = ImporterNode(
    instance_name='Schema_Importer',
    source_uri=SCHEMA_DIR,
    artifact_type=tfx.types.standard_artifacts.Schema,
    reimport=False)
context.run(schema_importer) ###RUN

statistics_gen = StatisticsGen(
      examples=example_gen.outputs['examples'],
      schema=schema_importer.outputs['result'],
      ).with_id('compute-eval-stats')
context.run(statistics_gen) ###RUN

### Example Validator - produce anamolies
example_validator = ExampleValidator(
    instance_name="Data_Validation",    
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_importer.outputs['result'])
context.run(example_validator) ###RUN
###context.show(example_validator.outputs['output']) shows any anomolies

### Transformer
TRANSFORM_MODULE = 'preprocessing.py'
transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_importer.outputs['result'],
    module_file=TRANSFORM_MODULE) #will call the preprocessing_fn
context.run(transform) ###RUN



### TRAINING!!!!
TRAINER_MODULE_FILE = 'model.py'
if HYPERTUNE:
    trainer = hypertune(context, transform, schema_importer, TRAINER_MODULE_FILE)
else:
    trainer = Trainer(
        custom_executor_spec=executor_spec.ExecutorClassSpec(trainer_executor.GenericExecutor),
        module_file=TRAINER_MODULE_FILE,
        transformed_examples=transform.outputs['transformed_examples'],
        schema=schema_importer.outputs['result'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=5000),
        eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=1000))
    context.run(trainer) ###RUN!

### Get the Best Model
model_resolver = ResolverNode(
      instance_name='latest_blessed_model_resolver',
      resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(type=ModelBlessing))
context.run(model_resolver) ###RUN!


### Analyze Best Model
accuracy_threshold = tfma.MetricThreshold(
                value_threshold=tfma.GenericValueThreshold(
                    lower_bound={'value': 0.5},
                    upper_bound={'value': 0.99})
                )

metrics_specs = tfma.MetricsSpec(
                   metrics = [
                       tfma.MetricConfig(class_name='SparseCategoricalAccuracy',
                           threshold=accuracy_threshold), ### given above
                       tfma.MetricConfig(class_name='ExampleCount')])

eval_config = tfma.EvalConfig(
    model_specs=[
        tfma.ModelSpec(label_key='Cover_Type')
    ],
    metrics_specs=[metrics_specs], ### given above
    slicing_specs=[
        tfma.SlicingSpec(),
        tfma.SlicingSpec(feature_keys=['Wilderness_Area'])
    ]
)

model_analyzer = Evaluator(
    examples=example_gen.outputs.examples,
    model=trainer.outputs.model,
    baseline_model=model_resolver.outputs.model,
    eval_config=eval_config
)
context.run(model_analyzer, enable_cache=False) ###RUN!


### Infra Validator
infra_validator = InfraValidator(
    model=trainer.outputs['model'],
    examples=example_gen.outputs['examples'],
    serving_spec=infra_validator_pb2.ServingSpec(
        tensorflow_serving=infra_validator_pb2.TensorFlowServing(
            tags=['latest']),
      local_docker=infra_validator_pb2.LocalDockerConfig(),
  ),
    validation_spec=infra_validator_pb2.ValidationSpec(
        max_loading_time_seconds=60,
        num_tries=5,
    ),    
  request_spec=infra_validator_pb2.RequestSpec(
      tensorflow_serving=infra_validator_pb2.TensorFlowServingRequestSpec(),
          num_examples=5,
      )
)
context.run(infra_validator, enable_cache=False) ###RUN!



### Pusher
pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=model_analyzer.outputs['blessing'],
    infra_blessing=infra_validator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=SERVING_MODEL_DIR)))
context.run(pusher) ### RUN!

