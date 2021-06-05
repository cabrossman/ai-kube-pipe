import os  # pylint: disable=unused-import

PIPELINE_NAME = 'demo_hackathon2020'
GCS_BUCKET_NAME = 'hostedkfp-default-68ypbzt0rw'
GCP_PROJECT_ID = 'trr-assignments-staging'
GCP_REGION = 'us-central1'

PREPROCESSING_FN = 'models.preprocessing.preprocessing_fn'
RUN_FN = 'models.keras.model.run_fn'
TRAIN_NUM_STEPS = 100
EVAL_NUM_STEPS = 100
EVAL_ACCURACY_THRESHOLD = 0.6
BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS = [
    '--project=' + GCP_PROJECT_ID,
    ]
_query_sample_rate = 0.0001  # Generate a 0.01% random sample.
#_query_sample_rate = 0.25
BIG_QUERY_QUERY = """
         SELECT
           pickup_community_area,
           fare,
           EXTRACT(MONTH FROM trip_start_timestamp) AS trip_start_month,
           EXTRACT(HOUR FROM trip_start_timestamp) AS trip_start_hour,
           EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS trip_start_day,
           UNIX_SECONDS(trip_start_timestamp) AS trip_start_timestamp,
           pickup_latitude,
           pickup_longitude,
           dropoff_latitude,
           dropoff_longitude,
           trip_miles,
           pickup_census_tract,
           dropoff_census_tract,
           payment_type,
           company,
           trip_seconds,
           dropoff_community_area,
           tips
         FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
         WHERE (ABS(FARM_FINGERPRINT(unique_key)) / 0x7FFFFFFFFFFFFFFF)
           < {query_sample_rate}""".format(
    query_sample_rate=_query_sample_rate)

DATAFLOW_BEAM_PIPELINE_ARGS = [
    '--project=' + GCP_PROJECT_ID,
    '--runner=DataflowRunner',
    '--temp_location=' + os.path.join('gs://', GCS_BUCKET_NAME, 'tmp'),
    '--region=' + GCP_REGION,
    '--experiments=shuffle_mode=auto',
    '--disk_size_gb=50',
    ]

GCP_AI_PLATFORM_TRAINING_ARGS = {
     'project': GCP_PROJECT_ID,
     'region': GCP_REGION,
     'masterConfig': {
       'imageUri': 'gcr.io/' + GCP_PROJECT_ID + '/tfx-pipeline'
     },
}

GCP_AI_PLATFORM_SERVING_ARGS = {
     'model_name': PIPELINE_NAME,
     'project_id': GCP_PROJECT_ID,
     'regions': [GCP_REGION],
}

