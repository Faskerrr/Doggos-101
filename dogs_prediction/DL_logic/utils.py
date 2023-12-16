import tensorflow as tf
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

def getImage(img=None, url_with_pic:str='', show=False):
    '''
    Get an image provided its url and resize it.
    The size of the image is 224x224.
    '''
    print(f"✅ get image received: img={img is not None}, url_with_pic={url_with_pic is not None}")
    if url_with_pic:
        response = requests.get(url_with_pic)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(img).convert("RGB")
    if show:
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    img = img.resize((224, 224))
    print("✅ image resized")
    return img

def compile_model(model):
    """
    Args:
        model: the model to compile
    Returns:
        model: the compiled model
    """
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("✅ Model compiled")
    return model

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
