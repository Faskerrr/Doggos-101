# registry.py is for loading and storing the model's weights from/to Cloud Storage

import glob
import os
import time
import pickle
from colorama import Fore, Style
from tensorflow import keras
from keras.models import load_model as keras_load_model
from dogs_prediction.params import *

# def save_model(model: keras.Model = None) -> None:
#     """
#     Persist trained model locally on hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
#     - if MODEL_TARGET='gcs', also persist it on your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
#     - if MODEL_TARGET='mlflow', also persist it on mlflow instead of GCS (for unit 0703 only) --> unit 03 only
#     """

#     # WE DO NOT NEED THIS BECAUSE WE ALREADY SAVED OUR MODELS IN models FOLDER
#     # timestamp = time.strftime("%Y%m%d-%H%M%S")
#     # # save model locally
#     # model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
#     # model.save(model_path)
#     # print("✅ Model saved locally")

#     model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", "Resnet_50_epoch.h5")  # need to be better #LOCAL_REGISTRY_PATH = ~/code/Faskerrr/Doggos-101

#     if MODEL_TARGET == "gcs":
#         from google.cloud import storage
#         model_filename = model_path.split("/")[-1]    # returns Resnet_50_epoch.h5
#         client = storage.Client()
#         bucket = client.bucket(BUCKET_NAME)
#         blob = bucket.blob(f"models/{model_filename}")
#         blob.upload_from_filename(model_path)

#         print("✅ Model saved to gcs")
#         return None

#     return None


def load_latest_model(loading_method = MODEL_TARGET):  # change to load_latest_model(stage="Production") if we want to use Mflow
    '''
    Function to load the latest model from local disk
    Requires to have a folder called "models" in the same directory
    '''
    # get the latest model from local
    # we need to create an .env file to store the path to the models folder
    if loading_method == "local":   # Change this to if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)
        local_model_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        print(LOCAL_REGISTRY_PATH)
        local_model_files = os.listdir(local_model_directory)
        local_model_paths = [os.path.join(local_model_directory, f) for f in local_model_files if f.endswith('.h5')]
        most_recent_model_path = max(local_model_paths, key=os.path.getctime)
        model_path = os.path.join(local_model_directory, most_recent_model_path)
        print("✅ Model loaded from local")
        latest_model = keras_load_model(model_path, compile = False)
        return latest_model
#local_model_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
    # get the latest model from GCP
    elif loading_method == "gcp":  # Change this to if MODEL_TARGET == "gcs":
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)
        from google.cloud import storage
        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="models"))       #get_bucket() retrieves a bucket via a GET request// list_blobs() returns an iterator used to find blobs in the bucket.
        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)    #LOCAL_REGISTRY_PATH = ~/code/Faskerrr/Doggos-101
            latest_blob.download_to_filename(latest_model_path_to_save)
            latest_model = keras_load_model(latest_model_path_to_save, compile = False)
            print("✅ Latest model downloaded from cloud storage")
            return latest_model
        except:
            print(f"\n❌ No model found on GCS bucket {BUCKET_NAME}")
            return None
