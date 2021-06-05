from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models.keras import model


class ModelTest(tf.test.TestCase):

  def testBuildKerasModel(self):
    built_model = model._build_keras_model(
        hidden_units=[1, 1], learning_rate=0.1)  # pylint: disable=protected-access
    self.assertEqual(len(built_model.layers), 8)

    built_model = model._build_keras_model(hidden_units=[1], learning_rate=0.1)  # pylint: disable=protected-access
    self.assertEqual(len(built_model.layers), 7)


if __name__ == '__main__':
  tf.test.main()
