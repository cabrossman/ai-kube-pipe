# Initial Setup

## Skaffold
### Linux - Set `PATH` to include user python binary directory and a directory containing `skaffold` if Linux.
```
PATH=$PATH:pwd/bin
```

### Mac
```
brew install skaffold
```

### Windows - :)

### Set some Variables
```
GCP_PROJECT_ID=trr-assignments-staging
CUSTOM_TFX_IMAGE=gcr.io/$GCP_PROJECT_ID/tfx-pipeline
PIPELINE_NAME=predict_conversion
ENDPOINT=XXXXXXXXXXXXXX #Get from GKE instance
```
### Navigate to pipeline
```
cd $PIPELINE_NAME
```

### Initial Setup for template - see pipeline_template. Skip for new versions/edits
```
tfx template copy \
 --pipeline-name=$PIPELINE_NAME \
 --destination-path=$PIPELINE_NAME \
 --model=taxi
 ```

 ### CREATE PIPELINE IMAGE (use this for first time)
 This will output a Dockerfile, build.yaml, and tar.gz
```
tfx pipeline create  \
 --pipeline-path=kubeflow_dag_runner.py \
 --endpoint=$ENDPOINT \
 --build-target-image=$CUSTOM_TFX_IMAGE
```

### Run Pipeline
View experiment by going to GKE cluster & to the exeriments nav
```
tfx run create --pipeline-name=$PIPELINE_NAME --endpoint=$ENDPOINT
```

 ### UPDATE Pipeline & Run (use after initial create) and run for each change
```
tfx pipeline update  \
 --pipeline-path=kubeflow_dag_runner.py \
 --endpoint=$ENDPOINT

tfx run create --pipeline-name=$PIPELINE_NAME --endpoint=$ENDPOINT
```

### Notes
kubeflow_dag_runn.py
```
All output files are stored under OUTPUT_DIR
TFX produces two types of outputs, files and metadata. - Files will be created under PIPELINE_ROOT directory.
SERVING_MODEL_DIR - is where the Pusher will produce serving model
DATA_PATH - data directory for CSV files for CsvExampleGen in pipeline.pipeline.CsvExampleGen - gs://bucket/chicago_taxi_trips/csv/
tfx_image - This pipeline automatically injects the Kubeflow TFX image if the environment variable 'KUBEFLOW_TFX_IMAGE' is defined. Currently, the tfx cli tool exports the environment variable to pass to the pipelines.
```



pipeline.configs.py
```
GCP_AI_PLATFORM_TRAINING_ARGS 
- for more info on args go to https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#Job

GCP_AI_PLATFORM_TRAINING_ARGS.masterConfig
- AI Platform uses custom containers : https://cloud.google.com/ml-engine/docs/containers-overview
- Note if a custom container, ensure the entrypoint calls TFX's run_executor script (tfx/scripts/run_executor.py)

GCP_AI_PLATFORM_SERVING_ARGS
- serving job parameters to be passed to AI platform
```

pipeline.pipeline.py
```
Look for components list - this executes each of the tasks run on Kubeflow

- [BQ/Csv]ExampleGen : brings data into pipeline
- StatisticsGen : Computes statistics over data for visualization and example validation.
- SchemaGen : Generates schema based on statistics files.
- ExampleValidator : Performs anomaly detection based on statistics and data schema.
- Transform : transformations for feature engineering
- Trainer : Trains Model
- ResolverNode : Get the latest blessed model for model validation.
- Evaluator : Uses TFMA to compute a evaluation statistics over features of a model and perform quality validation of a candidate model (compared to a baseline). Note change threshold will be ignored if there is no baseline (first run)
- Pusher : Checks whether the model passed the validation steps and pushes the model to a file dest
```

models.features.py
```
Need atleast one feature to work
DENSE_FLOAT_FEATURE_KEYS : Name of features which have continuous float values. These features will be used as their own values.
BUCKET_FEATURE_KEYS : features will be bucketized using `tft.bucketize` as categorical
BUCKET_FEATURE_BUCKET_COUNT : Number of buckets used by tf.transform for encoding each feature. Length of this list should be same with BUCKET_FEATURE_KEYS
CATEGORICAL_FEATURE_KEYS : Name of features which have categorical values which are mapped to integers. These features will be used as categorical features.
CATEGORICAL_FEATURE_MAX_VALUES : Number of buckets to use integer numbers as categorical features. The length of this list should be the same with CATEGORICAL_FEATURE_KEYS.
VOCAB_FEATURE_KEYS : Name of features which have string values and are mapped to integers.
VOCAB_SIZE : Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
OOV_SIZE : Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
```



models.preprocessing.py




models.keras.constants.py
```
constants used in models.keras.model.py
```

models.keras.model.py
