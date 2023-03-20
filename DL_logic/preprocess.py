import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

path = '/content/dogs/cropped/cropped/train'

def load_images(path:str):
  '''
  Load images from the Dog Dataset of Kaggle.
  Returns X and y as numpy arrays
  '''
  X = []
  y = []
  # Use os.walk to iterate over all the subdirectories and files within the given directory
  for subdir, dirs, files in os.walk(path):
      for file in tqdm(files):
          # Extract the label from the path
          # remove .split("-")[1] if you want to keep the ID
          label = os.path.basename(os.path.normpath(subdir)).split("-")[1]
          # Load the image using OpenCV
          img_path = os.path.join(subdir, file)
          img = cv2.imread(img_path)
          # Add the image data and label to the X and y arrays
          X.append(img)
          y.append(label)
  # Convert X and y to NumPy arrays and return them
  return np.array(X), np.array(y)


def load_images_and_preprocess(path):
    '''
    Load images from the Dog Dataset of Kaggle.
    Returns X and y as numpy arrays
    Encode labels as categorical variables
    '''
    X, y = load_images(path)
    assert X.shape == (12000, 224, 224, 3), f"Expected X to have shape (12000, 224, 224, 3), but got {X.shape}. Please check that you provided the correct path and that all images are loaded correctly."
    l_e = LabelEncoder()
    y = l_e.fit_transform(y)
    y = to_categorical(y)
    return X, y
