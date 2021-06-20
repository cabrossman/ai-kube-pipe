"""Covertype model features."""
import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

NUMERIC_FEATURE_KEYS = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points'
]

CATEGORICAL_FEATURE_KEYS = ['Wilderness_Area', 'Soil_Type']

LABEL_KEY = 'Cover_Type'
NUM_CLASSES = 7

def transformed_name(key):
  return key + '_xf'
