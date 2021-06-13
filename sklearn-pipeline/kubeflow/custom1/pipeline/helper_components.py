"""Helper components."""
from typing import NamedTuple, List


def retrieve_best_run(
    project_id: str, job_id: str
) -> NamedTuple('Outputs', [('metric_value', float), ('min_samples_leaf', int),
                            ('max_depth', int), ('max_features', int)]):
  """Retrieves the parameters of the best Hypertune run."""

  from googleapiclient import discovery
  from googleapiclient import errors

  ml = discovery.build('ml', 'v1')

  job_name = 'projects/{}/jobs/{}'.format(project_id, job_id)
  request = ml.projects().jobs().get(name=job_name)

  try:
    response = request.execute()
  except errors.HttpError as err:
    print(err)
  except:
    print('Unexpected error')

  print(response)

  best_trial = response['trainingOutput']['trials'][0]

  metric_value = best_trial['finalMetric']['objectiveValue']
  min_samples_leaf = int(best_trial['hyperparameters']['min_samples_leaf'])
  max_depth = int(best_trial['hyperparameters']['max_depth'])
  max_features = int(best_trial['hyperparameters']['max_features'])

  return (metric_value, min_samples_leaf, max_depth, max_features)


def evaluate_model(
    dataset_path: str, model_path: str, metric_name: str
) -> NamedTuple('Outputs', [('metric_name', str), ('metric_value', float),
                            ('mlpipeline_metrics', 'Metrics')]):
  """Evaluates a trained sklearn model."""
  #import joblib
  import pickle
  import json
  import pandas as pd
  import subprocess
  import sys

  from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score

  df_test = pd.read_csv(dataset_path)

  X_test = df_test.drop('sold', axis=1)
  y_test = df_test['sold']

  # Copy the model from GCS
  model_filename = 'model.pkl'
  gcs_model_filepath = '{}/{}'.format(model_path, model_filename)
  print(gcs_model_filepath)
  subprocess.check_call(['gsutil', 'cp', gcs_model_filepath, model_filename],
                        stderr=sys.stdout)

  with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

  y_hat = model.predict(X_test)

  if metric_name == 'accuracy':
    metric_value = balanced_accuracy_score(y_test, y_hat)
  elif metric_name == 'recall':
    metric_value = recall_score(y_test, y_hat)
  elif metric_name == 'precision_score':
    metric_value = precision_score(y_test, y_hat)
  else:
    metric_name = 'N/A'
    metric_value = 0

  # Export the metric
  metrics = {
      'metrics': [{
          'name': metric_name,
          'numberValue': float(metric_value)
      }]
  }

  return (metric_name, metric_value, json.dumps(metrics))


def overwrite_production_model(root: str, input_path: str, model_id: str, version: str
) -> NamedTuple('Outputs', [('destination_blob_name', str)]):
    from google.cloud import storage
    """Copies a blob from one bucket to another with a new name."""

    bucket_name = root.split('/')[2]

    tmp = '/'.join(input_path.split(bucket_name)[1].split('/')[1:])
    blob_name = '{}/model.pkl'.format(tmp)

    destination_bucket_name = 'kubeflow-prod-bucket-ex'
    destination_blob_name = 'pricing/model/{}/version/{}/model.pkl'.format(model_id, version)

    storage_client = storage.Client()

    source_bucket = storage_client.bucket(bucket_name)
    source_blob = source_bucket.blob(blob_name)
    destination_bucket = storage_client.bucket(destination_bucket_name)

    blob_copy = source_bucket.copy_blob(
        source_blob, destination_bucket, destination_blob_name
    )

    print(
        "Blob {} in bucket {} copied to blob {} in bucket {}.".format(
            source_blob.name,
            source_bucket.name,
            blob_copy.name,
            destination_bucket.name,
        )
    )

    return (destination_blob_name)


def get_split_q(source_table_name: str, num_lots: int, lots: List):
    from jinja2 import Template
    """Prepares the data sampling query."""

    sampling_query_template = """
         SELECT
              latest_customer_price,
              latest_promotion_rate,
              percent_available,
              latest_base_price,
              original_price,
              total,
              item_views,
              items_sold,
              average_retail,
              designer_id,
              taxon_permalink,
              condition,
              merchandising_category,
              case when change_type = 'sale' then 1 else 0 end as sold
         FROM 
             `{{ source_table }}` AS cover
         WHERE 
          MOD(ABS(FARM_FINGERPRINT(cast(item_id as STRING))), {{ num_lots }}) IN ({{ lots }})
          and DATE(available_at) < DATE('2021-05-01')
          and DATE(available_at) > DATE('2021-01-01')
          and change_type in ('discount','auto','sale')
          and item_state = 'launched'
         """

    query = Template(sampling_query_template).render(
        source_table=source_table_name, num_lots=num_lots, lots=str(lots)[1:-1])

    return query

