# registry.py is for loading and storing the model's weights from/to Cloud Storage

import glob
import os
import time
import pickle
from colorama import Fore, Style
from tensorflow import keras
from keras.models import load_model as keras_load_model
from dogs_prediction.params import *


def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it on your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on mlflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    # WE DO NOT NEED THIS BECAUSE WE ALREADY SAVED OUR MODELS IN models FOLDER
    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    # # save model locally
    # model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    # model.save(model_path)
    # print("✅ Model saved locally")

    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", "Resnet_50_epoch.h5")  # need to be better #LOCAL_REGISTRY_PATH = ~/code/Faskerrr/Doggos-101

    if MODEL_TARGET == "gcs":
        from google.cloud import storage
        model_filename = model_path.split("/")[-1]    # returns Resnet_50_epoch.h5
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to gcs")
        return None

    return None


# FUNCTION FROM TAXI FARE (Jihed has the load_latest_model(loading_method). Do we need this function here? Should we move Jihed's function to here?)
def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model found

    """
    # LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)
        # Get latest model version name by timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")
        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)
        lastest_model = keras.models.load_model(most_recent_model_path_on_disk)
        print("✅ model loaded from local disk")

        return lastest_model

    elif MODEL_TARGET == "gcs":
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        from google.cloud import storage
        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))    #get_bucket() retrieves a bucket via a GET request// list_blobs() returns an iterator used to find blobs in the bucket.
        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)
            latest_model = keras.models.load_model(latest_model_path_to_save)
            print("✅ Latest model downloaded from cloud storage")
            return latest_model
        except:
            print(f"\n❌ No model found on GCS bucket {BUCKET_NAME}")
            return None


# JIHED's FUNCTION
def load_latest_model(loading_method):  # change to load_latest_model(stage="Production") -> keras.Model:  ???
    '''
    Function to load the latest model from local disk
    Requires to have a folder called "models" in the same directory
    '''
    # get the latest model from local
    # we need to create an .env file to store the path to the models folder
    if loading_method == "local":   # Change this to if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)
        local_model_directory = os.path.join(os.getcwd(), '../models') # this needs to be improved!
        local_model_files = os.listdir(local_model_directory)
        local_model_paths = [os.path.join(local_model_directory, f) for f in local_model_files if f.endswith('.h5')]
        most_recent_model_path = max(local_model_paths, key=os.path.getctime)
        model_path = os.path.join(local_model_directory, most_recent_model_path)
        print("✅ Model loaded from local")
        latest_model = keras_load_model(model_path, compile = False)
        return latest_model

    # get the latest model from GCP
    elif loading_method == "gcp":  # Change this to if MODEL_TARGET == "gcs":
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)
        from google.cloud import storage
        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))
        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)    #LOCAL_REGISTRY_PATH = ~/code/Faskerrr/Doggos-101
            latest_blob.download_to_filename(latest_model_path_to_save)
            latest_model = keras.models.load_model(latest_model_path_to_save)
            print("✅ Latest model downloaded from cloud storage")
            latest_model = keras_load_model(model_path, compile = False)
            return latest_model
        except:
            print(f"\n❌ No model found on GCS bucket {BUCKET_NAME}")
            return None
