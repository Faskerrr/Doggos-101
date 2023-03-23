# params.py contains the project's global variables/parameters (including variables from .env)

import os
import numpy as np

# WE NEED TO CREATE a .env file

##################  VARIABLES  ##################
MODEL_TARGET = os.environ.get("MODEL_TARGET")    # get MODEL_TARGET from .env (e.g. local, gcs, mflow)
GCP_PROJECT = os.environ.get("GCP_PROJECT")      # get GCP_PROJECT from .env (personal GCP project for this bootcamp)
BUCKET_NAME = os.environ.get("BUCKET_NAME")      # get BUCKET_NAME from .env (cloud storage)
INSTANCE = os.environ.get("INSTANCE")            # get INSTANCE from .env (name of our Virtual Machine instance) NOT SURE IF WE NEED


##################  CONSTANTS  #####################
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "code", "Faskerrr", "Doggos-101")
#LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")
# EXAMPLE: .lewagon\mlops\training_outputs" (metrics, models, params)





################## VALIDATIONS #################
env_valid_options = dict(
    MODEL_TARGET=["local", "gcs"],
)


#JIHED's code (same as os.path.join(LOCAL_REGISTRY_PATH, "models", "Resnet_50_epoch.h5")
local_model_directory = os.path.join(os.getcwd(), '../models') # this needs to be improved!
local_model_files = os.listdir(local_model_directory)
local_model_paths = [os.path.join(local_model_directory, f) for f in local_model_files if f.endswith('.h5')]
most_recent_model_path = max(local_model_paths, key=os.path.getctime)
model_path = os.path.join(local_model_directory, most_recent_model_path)
print("✅ Model loaded from local")
#'/home/lilla/code/Faskerrr/Doggos-101/models/Resnet_50_epoch.h5'
