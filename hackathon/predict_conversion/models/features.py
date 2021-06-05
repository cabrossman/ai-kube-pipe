from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text, List

DENSE_FLOAT_FEATURE_KEYS = [
  'session_duration_seconds', 'tracks_count', 'pages_count', 'product_viewed', 'product_added', 'product_list_viewed', 'product_list_filtered'
]
BUCKET_FEATURE_KEYS = [

]
BUCKET_FEATURE_BUCKET_COUNT = [

]
CATEGORICAL_FEATURE_KEYS = [
  'session_start_month', 'session_start_hour', 'session_start_day', 'session_type', 'visited_homepage', 'customer_type'
]
CATEGORICAL_FEATURE_MAX_VALUES = [
  31, 24, 31, 3, 2, 3
]
VOCAB_FEATURE_KEYS = [
  'campaign_name', 'campaign_source', 'campaign_medium', 'refr_host', 'source'
]
VOCAB_SIZE = 250
OOV_SIZE = 10

LABEL_KEY = 'converted'


def transformed_name(key: Text) -> Text:
  """Generate the name of the transformed feature from original name."""
  return key + '_xf'


def vocabulary_name(key: Text) -> Text:
  """Generate the name of the vocabulary feature from original name."""
  return key + '_vocab'


def transformed_names(keys: List[Text]) -> List[Text]:
  """Transform multiple feature names at once."""
  return [transformed_name(key) for key in keys]
