#https://github.com/GoogleCloudPlatform/mlops-on-gcp/blob/master/examples/tfdv-structured-data/tfdv-covertype.ipynb

import os
import tempfile
import tensorflow as tf
import tensorflow_data_validation as tfdv
import time

from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions, StandardOptions, SetupOptions, DebugOptions, WorkerOptions
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2, statistics_pb2

print('TensorFlow version: {}'.format(tf.__version__))
print('TensorFlow Data Validation version: {}'.format(tfdv.__version__))


# Set GCS data
TRAINING_DATASET='gs://workshop-datasets/covertype/training/dataset.csv'
TRAINING_DATASET_WITH_MISSING_VALUES='gs://workshop-datasets/covertype/training_missing/dataset.csv'
DATA_ROOT = 'gs://workshop-datasets/covertype/small/dataset.csv'
EVALUATION_DATASET='gs://workshop-datasets/covertype/evaluation/dataset.csv'
EVALUATION_DATASET_WITH_ANOMALIES='gs://workshop-datasets/covertype/evaluation_anomalies/dataset.csv'
SERVING_DATASET='gs://workshop-datasets/covertype/serving/dataset.csv'

os.chdir('/Users/christopher.brossman/vs/tuts/kubeflow/ai-kube-pipe/tfx/tfdv/lab')
BASE_DIR = os.path.join(os.sep, os.getcwd(),'artifact-store')

PROJECT_ID = 'mlops-workshop'
REGION = 'us-central1'
STAGING_BUCKET = 'gs://{}-staging'.format(PROJECT_ID)


#get stats
train_stats = tfdv.generate_statistics_from_csv(
    data_location=DATA_ROOT
)
schema = tfdv.infer_schema(train_stats)

tfdv.get_feature(schema, 'Soil_Type').type = schema_pb2.FeatureType.BYTES
tfdv.set_domain(schema, 'Soil_Type', schema_pb2.StringDomain(name='Soil_Type', value=[]))
tfdv.get_domain(schema, 'Soil_Type').value.append('5151')


tfdv.set_domain(schema, 'Cover_Type', schema_pb2.IntDomain(name='Cover_Type', min=1, max=7, is_categorical=True))

tfdv.get_feature(schema, 'Slope').type = schema_pb2.FeatureType.FLOAT
tfdv.set_domain(schema, 'Slope',  schema_pb2.FloatDomain(name='Slope', min=0, max=90))


domain = tfdv.utils.schema_util.schema_pb2.IntDomain(
    min=0, 
    max=300
)
tfdv.set_domain(schema, 'Hillshade_Noon', domain)

