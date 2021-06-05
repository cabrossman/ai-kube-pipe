from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models import features


class FeaturesTest(tf.test.TestCase):

  def testNumberOfBucketFeatureBucketCount(self):
    self.assertEqual(
        len(features.BUCKET_FEATURE_KEYS),
        len(features.BUCKET_FEATURE_BUCKET_COUNT))
    self.assertEqual(
        len(features.CATEGORICAL_FEATURE_KEYS),
        len(features.CATEGORICAL_FEATURE_MAX_VALUES))

  def testTransformedNames(self):
    names = ["f1", "cf"]
    self.assertEqual(["f1_xf", "cf_xf"], features.transformed_names(names))


if __name__ == "__main__":
  tf.test.main()
