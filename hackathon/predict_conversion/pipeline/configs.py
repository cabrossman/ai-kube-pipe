import os

PIPELINE_NAME = 'predict_conversion'
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

BIG_QUERY_QUERY = """
  SELECT 
    EXTRACT(MONTH FROM session_start_at) AS session_start_month,
    EXTRACT(HOUR FROM session_start_at) AS session_start_hour,
    EXTRACT(DAYOFWEEK FROM session_start_at) AS session_start_day,
    UNIX_SECONDS(session_end_at) - UNIX_SECONDS(session_start_at) as session_duration_seconds,
    tracks_count,
    pages_count,
    product_viewed,
    product_added,
    product_list_viewed,
    product_list_filtered,
    cast(repeat as INT64) AS repeat,
    session_type,
    back_end,
    cast(visited_homepage as INT64) AS visited_homepage,
    LOWER(TRIM(campaign_name)) AS campaign_name,
    LOWER(TRIM(campaign_source)) AS campaign_source,
    LOWER(TRIM(campaign_medium)) AS campaign_medium,
    LOWER(TRIM(refr_host)) AS refr_host,
    LOWER(TRIM(sid)) AS sid,
    LOWER(TRIM(source)) AS source,
    LOWER(TRIM(sub_source)) AS sub_source,
    customer_type,
    cast(converted as INT64) AS converted

  FROM `trr-assignments-staging.hackathon.trr_desktop_sessions_q2_dev`
  WHERE converted IS NOT NULL
"""

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

