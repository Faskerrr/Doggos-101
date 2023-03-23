# Import libraries
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os
from keras.models import load_model as keras_load_model

from tensorflow.keras import optimizers
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess_input

# shall we make a environment variable for this?
# we can then call it this way
# breed = os.environ.get('breed')
breed = ['Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih', 'Blenheim_spaniel', 'papillon', 'toy_terrier', 'Rhodesian_ridgeback', 'Afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick', 'black', 'Walker_hound', 'English_foxhound', 'redbone', 'borzoi', 'Irish_wolfhound', 'Italian_greyhound', 'whippet', 'Ibizan_hound', 'Norwegian_elkhound', 'otterhound', 'Saluki', 'Scottish_deerhound', 'Weimaraner', 'Staffordshire_bullterrier', 'American_Staffordshire_terrier', 'Bedlington_terrier', 'Border_terrier', 'Kerry_blue_terrier', 'Irish_terrier', 'Norfolk_terrier', 'Norwich_terrier', 'Yorkshire_terrier', 'wire', 'Lakeland_terrier', 'Sealyham_terrier', 'Airedale', 'cairn', 'Australian_terrier', 'Dandie_Dinmont', 'Boston_bull', 'miniature_schnauzer', 'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier', 'Tibetan_terrier', 'silky_terrier', 'soft', 'West_Highland_white_terrier', 'Lhasa', 'flat', 'curly', 'golden_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever', 'German_short', 'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter', 'Brittany_spaniel', 'clumber', 'English_springer', 'Welsh_springer_spaniel', 'cocker_spaniel', 'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old_English_sheepdog', 'Shetland_sheepdog', 'collie', 'Border_collie', 'Bouvier_des_Flandres', 'Rottweiler', 'German_shepherd', 'Doberman', 'miniature_pinscher', 'Greater_Swiss_Mountain_dog', 'Bernese_mountain_dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull_mastiff', 'Tibetan_mastiff', 'French_bulldog', 'Great_Dane', 'Saint_Bernard', 'Eskimo_dog', 'malamute', 'Siberian_husky', 'affenpinscher', 'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Great_Pyrenees', 'Samoyed', 'Pomeranian', 'chow', 'keeshond', 'Brabancon_griffon', 'Pembroke', 'Cardigan', 'toy_poodle', 'miniature_poodle', 'standard_poodle', 'Mexican_hairless', 'dingo', 'dhole', 'African_hunting_dog']

def load_latest_model(loading_method):
    '''
    Function to load the latest model from local disk
    Requires to have a folder called "models" in the same directory
    '''
    # get the latest model from local
    # we need to create an .env file to store the path to the models folder
    if loading_method == "local":
        local_model_directory = os.path.join(os.getcwd(), '../models') # this needs to be improved!
        local_model_files = os.listdir(local_model_directory)
        local_model_paths = [os.path.join(local_model_directory, f) for f in local_model_files if f.endswith('.h5')]
        most_recent_model_path = max(local_model_paths, key=os.path.getctime)
        model_path = os.path.join(local_model_directory, most_recent_model_path)
        print("✅ Model loaded from local")

    # get the latest model from GCP
    elif loading_method == "gcp":
        model_path = "TO BE ADDED LATER"
        print("✅ Model loaded from the cloud")

    latest_model = keras_load_model(model_path, compile = False)
    return latest_model

def compile_model(model):
    """
    Compile the model.
    Args:
        model: the model to compile
    Returns:
        model: the compiled model
    """
    opt = optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("✅ Model compiled")
    return model

def getImage(url:str='', pic=None):
  '''
  Get an image provided its url and resize it.
  The size of the image is 224x224.
  '''
  if url:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
  else:
    img = Image.open(pic)
  plt.imshow(img)
  plt.axis('off')
  plt.show()
  img = img.resize((224, 224))
  return img

# Check if this works with Inception
# We could also change the name of the function
def predict_labels(model, model_type, *args, **kwargs):
    '''
    Function that will load the latest model from local disk and use it to predict the breed of the dog in the image.
    Args:
        url: url of the image to predict
    Returns:
        breed_prediction: dictionary with the top 3 breeds predicted
        score_prediction: dictionary with the top 3 scores predicted
    '''
    img = getImage(*args, **kwargs)
    print("✅ Image successfully loaded")
    img = img_to_array(img)    #shape = (224, 224, 3)
    img = img.reshape((-1, 224, 224, 3))
    print("✅ Image successfully reshaped", img.shape)

    model = compile_model(model)
    print("✅ Model successfully loaded and compiled")

    if model_type == "resnet50":
        img = resnet_preprocess_input(img)
        print("✅ Image successfully preprocessed (resnet50)")
    elif model_type == "inception_v3":
        img = inception_preprocess_input(img)
        print("✅ Image successfully preprocessed (inception_v3)")
    # img = resnet_preprocess_input(img)
    # print("✅ Image successfully preprocessed ")
    print("✅ Predicting breed...")

    res = model.predict(img)
    print("✅ Breed predicted")
    indexes = np.argsort(res)[0][-3:][::-1]
    predicts = np.sort(res)[0][::-1][0:3]

    breed_prediction = {
        'first': breed[indexes[0]],
        'second': breed[indexes[1]],
        'third': breed[indexes[2]],
    }
    score_prediction = {
        'first': float(round(predicts[0],2)),
        'second': float(round(predicts[1],2)),
        'third': float(round(predicts[2],2))
    }
    output = {'prediction': breed_prediction,
              'score': score_prediction}
    return  output
