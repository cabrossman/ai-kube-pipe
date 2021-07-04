#https://www.tensorflow.org/tfx/data_validation/get_started

# https://cloud.google.com/architecture/analyzing-and-validating-data-at-scale-for-ml-using-tfx

# https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto

# https://github.com/GoogleCloudPlatform/mlops-on-gcp/tree/master/examples/tfdv-structured-data


import os
import tempfile, urllib, zipfile

# Set up some globals for our file paths
os.chdir('/Users/christopher.brossman/vs/tuts/kubeflow/ai-kube-pipe/tfx/tfdv/lab')
BASE_DIR = os.path.join(os.sep, os.getcwd(),'artifact-store')
SCHEMA_DIR = os.path.join(BASE_DIR, 'schema')
os.makedirs(SCHEMA_DIR, exist_ok=True)
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'chicago_taxi_output')
TRAIN_DATA = os.path.join(DATA_DIR, 'train', 'data.csv')
EVAL_DATA = os.path.join(DATA_DIR, 'eval', 'data.csv')
SERVING_DATA = os.path.join(DATA_DIR, 'serving', 'data.csv')

# Download the zip file from GCP and unzip it
zip, headers = urllib.request.urlretrieve('https://storage.googleapis.com/artifacts.tfx-oss-public.appspot.com/datasets/chicago_data.zip')
zipfile.ZipFile(zip).extractall(BASE_DIR)
zipfile.ZipFile(zip).close()

print("Here's what we downloaded:")
!ls -R {os.path.join(BASE_DIR, 'data')}


import tensorflow_data_validation as tfdv
print('TFDV version: {}'.format(tfdv.version.__version__))


train_stats = tfdv.generate_statistics_from_csv(data_location=TRAIN_DATA)
##tfdv.visualize_statistics(train_stats) -> This generates HTML 

schema = tfdv.infer_schema(statistics=train_stats)
SCHEMA_FILE = os.path.join(SCHEMA_DIR, 'schema.pbtxt')
tfdv.write_schema_text(schema, SCHEMA_FILE)
#tfdv.display_schema(schema=schema) -> display schema
# Outputs
"""
schema.annotation

schema.feature
schema.dataset_constraints
schema.default_environment
schema.weighted_feature
schema.tensor_representation_group
schema.sparse_feature

#domains
schema.float_domain
schema.int_domain
schema.string_domain


"""


"""So far we've only been looking at the training data. 
It's important that our evaluation data is consistent with our training data, 
including that it uses the same schema. It's also important that the evaluation data 
includes examples of roughly the same ranges of values for our numerical features 
as our training data, so that our coverage of the loss surface during evaluation 
is roughly the same as during training. The same is true for categorical features. 
Otherwise, we may have training issues that are not identified during evaluation, 
because we didn't evaluate part of our loss surface."""


# Compute stats for evaluation data
eval_stats = tfdv.generate_statistics_from_csv(data_location=EVAL_DATA)

# Compare evaluation data with training data -> generates HTML with colors to visualize diff
#tfdv.visualize_statistics(lhs_statistics=eval_stats, rhs_statistics=train_stats,
#                          lhs_name='EVAL_DATASET', rhs_name='TRAIN_DATASET')


"""
Check for evaluation anomalies
Does our evaluation dataset match the schema from our training dataset? 
This is especially important for categorical features, where we want to identify 
the range of acceptable values.

Key Point: What would happen if we tried to evaluate using data with 
categorical feature values that were not in our training dataset? 
What about numeric features that are outside the ranges in our training dataset?
"""
anomalies = tfdv.validate_statistics(statistics=eval_stats, schema=schema)
tfdv.write_anomalies_text(anomalies, '{}/anomalies.pbtxt'.format(SCHEMA_DIR))
print(anomalies.anomaly_info)


# Relax the minimum fraction of values that must come from the domain for feature company.
company = tfdv.get_feature(schema, 'company')
company.distribution_constraints.min_domain_mass = 0.9
"""
# Feature methods
annotation
bool_domain
deprecated
distribution_constraints
domain
drift_comparator
float_domain
group_presence
image_domain
in_environment
int_domain
lifecycle_stage
mid_domain
name
natural_language_domain
not_in_environment
presence
shape
skew_comparator
string_domain
struct_domain
time_domain
time_of_day_domain
type
unique_constraints
url_domain
value_count
value_counts
"""

# Add new value to the domain of feature payment_type.
payment_type_domain = tfdv.get_domain(schema, 'payment_type')
payment_type_domain.value.append('Prcard')
"""
name
value
"""

# Validate eval stats after updating the schema 
updated_anomalies = tfdv.validate_statistics(eval_stats, schema)
#tfdv.display_anomalies(updated_anomalies)
print(updated_anomalies.anomaly_info)


serving_stats = tfdv.generate_statistics_from_csv(SERVING_DATA)
serving_anomalies = tfdv.validate_statistics(serving_stats, schema)
#tfdv.display_anomalies(serving_anomalies)
print(serving_anomalies.anomaly_info)


options = tfdv.StatsOptions(schema=schema, infer_type_from_schema=True)
serving_stats = tfdv.generate_statistics_from_csv(SERVING_DATA, stats_options=options)
serving_anomalies = tfdv.validate_statistics(serving_stats, schema)
tfdv.display_anomalies(serving_anomalies)


# All features are by default in both TRAINING and SERVING environments.
schema.default_environment.append('TRAINING')
schema.default_environment.append('SERVING')

# Specify that 'tips' feature is not in SERVING environment.
tfdv.get_feature(schema, 'tips').not_in_environment.append('SERVING')

serving_anomalies_with_env = tfdv.validate_statistics(
    serving_stats, schema, environment='SERVING')

#tfdv.display_anomalies(serving_anomalies_with_env)
print(serving_anomalies_with_env.anomaly_info)


# Add skew comparator for 'payment_type' feature.
payment_type = tfdv.get_feature(schema, 'payment_type')
payment_type.skew_comparator.infinity_norm.threshold = 0.01

# Add drift comparator for 'company' feature.
company=tfdv.get_feature(schema, 'company')
company.drift_comparator.infinity_norm.threshold = 0.001

skew_anomalies = tfdv.validate_statistics(train_stats, schema,
                                          previous_statistics=eval_stats,
                                          serving_statistics=serving_stats)

#tfdv.display_anomalies(skew_anomalies)
print(skew_anomalies.anomaly_info)

# write updated schema
SCHEMA_FILE = os.path.join(SCHEMA_DIR, 'schema_updated.pbtxt')
tfdv.write_schema_text(schema, SCHEMA_FILE)