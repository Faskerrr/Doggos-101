# from tensorflow.keras import layers
import tensorflow as tf

class LayerScale(tf.keras.layers.Layer):
  """Layer scale module.

  Reqired for compiling the ConvNeXtXLargeNew model

  Reference:
    - hhttps://arxiv.org/abs/2103.17239

  Args:
    init_values (float): Initial value for layer scale. Should be within
      [0, 1].
    projection_dim (int): Projection dimensionality.

  Returns:
    Tensor multiplied to the scale.

  """
  def __init__(self, init_values, projection_dim, **kwargs):
    super().__init__(**kwargs)
    self.init_values = init_values
    self.projection_dim = projection_dim
    self.gamma = tf.Variable(self.init_values * tf.ones((self.projection_dim,)))

  def call(self, x):
    return x * self.gamma

  def get_config(self):
    config = super().get_config()
    config.update(
      {"init_values": self.init_values, "projection_dim": self.projection_dim}
    )
    return config
