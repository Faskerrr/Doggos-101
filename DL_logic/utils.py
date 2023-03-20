import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

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
