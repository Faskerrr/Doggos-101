# Functions to predict the breed of a dog given an image
# The images needs to be resized to 224x224
# The images need to be preprocess to be in the same format as the training images
# The model needs to be loaded
# The prediction needs to be made

# Path: dogs_prediction/DL_logic/predict.py

# Import libraries
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np



breed = ['Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih', 'Blenheim_spaniel', 'papillon', 'toy_terrier', 'Rhodesian_ridgeback', 'Afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick', 'black', 'Walker_hound', 'English_foxhound', 'redbone', 'borzoi', 'Irish_wolfhound', 'Italian_greyhound', 'whippet', 'Ibizan_hound', 'Norwegian_elkhound', 'otterhound', 'Saluki', 'Scottish_deerhound', 'Weimaraner', 'Staffordshire_bullterrier', 'American_Staffordshire_terrier', 'Bedlington_terrier', 'Border_terrier', 'Kerry_blue_terrier', 'Irish_terrier', 'Norfolk_terrier', 'Norwich_terrier', 'Yorkshire_terrier', 'wire', 'Lakeland_terrier', 'Sealyham_terrier', 'Airedale', 'cairn', 'Australian_terrier', 'Dandie_Dinmont', 'Boston_bull', 'miniature_schnauzer', 'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier', 'Tibetan_terrier', 'silky_terrier', 'soft', 'West_Highland_white_terrier', 'Lhasa', 'flat', 'curly', 'golden_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever', 'German_short', 'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter', 'Brittany_spaniel', 'clumber', 'English_springer', 'Welsh_springer_spaniel', 'cocker_spaniel', 'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old_English_sheepdog', 'Shetland_sheepdog', 'collie', 'Border_collie', 'Bouvier_des_Flandres', 'Rottweiler', 'German_shepherd', 'Doberman', 'miniature_pinscher', 'Greater_Swiss_Mountain_dog', 'Bernese_mountain_dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull_mastiff', 'Tibetan_mastiff', 'French_bulldog', 'Great_Dane', 'Saint_Bernard', 'Eskimo_dog', 'malamute', 'Siberian_husky', 'affenpinscher', 'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Great_Pyrenees', 'Samoyed', 'Pomeranian', 'chow', 'keeshond', 'Brabancon_griffon', 'Pembroke', 'Cardigan', 'toy_poodle', 'miniature_poodle', 'standard_poodle', 'Mexican_hairless', 'dingo', 'dhole', 'African_hunting_dog']

def predict_breed(url:str, model):
    img = getImage(url)
    img = img_to_array(img)    #shape = (224, 224, 3)
    img = img.reshape((-1, 224, 224, 3))   #shape=(1, 224, 224, 3)
    img = model.preprocess_input(img)
    res = model.predict(img)

    indexes = np.argsort(res)[0][-3:][::-1]      #.argmax return the indices of the maximum values along an axis.
    first = indexes[0]
    second = indexes[1]
    third = indexes[2]

    predicts = np.sort(res)[0][::-1][0:3]
    predict_first = round(predicts[0],2)
    predict_second = round(predicts[1],2)
    predict_third = round(predicts[2],2)

    #class_names = train_ds.class_names
    #class_names = [re.findall('n\d{8}-(.*)', i)[0].capitalize() for i in class_names]
    class_names = breed

    print(f"Top three breeds: {class_names[first]}, {class_names[second]}, {class_names[third]}")
    print(f"Top three probabilities = {predict_first*100} %, {predict_second*100} %, {predict_third*100} %")



import os
import glob
from keras.models import load_model as keras_load_model

def load_latest_model():
    '''
    Function to load the latest model from local disk
    Requires to have a folder called "models" in the same directory
    '''
    local_model_directory = os.path.join(os.getcwd(), 'models')
    print(local_model_directory)
    breakpoint()
    local_model_paths = glob.glob(f"{local_model_directory}/*.h5")
    breakpoint()
    print(local_model_paths)
    if not local_model_paths:
        return None
    # get the most recent model
    most_recent_model_path = max(local_model_paths, key=os.path.getctime)
    print(f"Loading model from {most_recent_model_path}")
    latest_model = keras_load_model(most_recent_model_path)
    print("✅ model loaded from local disk")
    return latest_model



def getImage(url:str):
  '''
  Get an image provided its url and resize it.
  The size of the image is 224x224.
  '''
  response = requests.get(url)
  img = Image.open(BytesIO(response.content))
  plt.imshow(img)
  img = img.resize((224, 224))
  return img

def ResNet50_predict_labels(url:str):
    '''
    Function that will load the latest model from local disk and use it to predict the breed of the dog in the image.
    Args:
        url: url of the image to predict
    Returns:
        breed_prediction: dictionary with the top 3 breeds predicted
        score_prediction: dictionary with the top 3 scores predicted
    '''
    img = getImage(url)
    print("✅ Image successfully loaded")
    img = img_to_array(img)    #shape = (224, 224, 3)
    img = img.reshape((-1, 224, 224, 3))
    print("✅ Image successfully reshaped", img.shape)
    model = load_model()
    img = model.preprocess_input(img)
    print("✅ Image successfully preprocessed")
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
        'first': round(predicts[0],2),
        'second': round(predicts[1],2),
        'third': round(predicts[2],2)
    }
    return  breed_prediction, score_prediction
