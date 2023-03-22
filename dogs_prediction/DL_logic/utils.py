import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img



def display_images(breed:str):
  '''
  Show an image of a breed if the breed is part of the classification.
  Adapt the path later.

  '''
  path = '/content/dogs/cropped/cropped/train'
  # Use os.walk to iterate over all the subdirectories and files within the given directory
  for subdir, dirs, files in os.walk(path):
      for file in tqdm(files):
          # Extract the label from the path
          # remove .split("-")[1] if you want to keep the ID
          label = os.path.basename(os.path.normpath(subdir)).split("-")[1]
          img_path = os.path.join(subdir, file)
          if breed == label:
            img = cv2.imread(img_path)
            plt.imshow(img)
            plt.axis('off')
            plt.xlabel(label)
            plt.show()


# Function for plotting the loss and accuracy
def plot_history(history, title='', axs=None, exp_name=""):
    '''
    Plot the loss and accuracy of a model over epochs.
    Args:
        history: the history object returned by model.fit
        title: title of the plot
        axs: axes to plot on
        exp_name: name of the experiment
    Returns:

    '''
    if axs is not None:
        ax1, ax2 = axs
    else:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if len(exp_name) > 0 and exp_name[0] != '_':
        exp_name = '_' + exp_name
    ax1.plot(history.history['loss'], label='train' + exp_name)
    ax1.plot(history.history['val_loss'], label='val' + exp_name)
    #ax1.set_ylim(0., 2.2)
    ax1.set_title('loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='train accuracy'  + exp_name)
    ax2.plot(history.history['val_accuracy'], label='val accuracy'  + exp_name)
    #ax2.set_ylim(0.25, 1.)
    ax2.set_title('Accuracy')
    ax2.legend()
    return (ax1, ax2)

def getImage(url):
  '''
  Get an image provided its url and resize it.
  The size of the image is 224x224.
  '''
  response = requests.get(url)
  img = Image.open(BytesIO(response.content))
  plt.imshow(img)
  img = img.resize((224, 224))
  return img
