# params.py contains the project's global variables/parameters (including variables from .env)
import os

##################  VARIABLES  ##################
MODEL_TARGET = os.environ.get("MODEL_TARGET")    # get MODEL_TARGET from .env (e.g. local, gcs, mflow)
GCP_PROJECT = os.environ.get("GCP_PROJECT")      # get GCP_PROJECT from .env (personal GCP project for this bootcamp)
BUCKET_NAME = os.environ.get("BUCKET_NAME")      # get BUCKET_NAME from .env (cloud storage)
INSTANCE = os.environ.get("INSTANCE")            # get INSTANCE from .env (name of our Virtual Machine instance) NOT SURE IF WE NEED
##################  CONSTANTS  #####################
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "code", "Faskerrr", "Doggos-101")    # ~code/Faskerrr/Doggos-101

##################  DOG BREEDS  #####################
BREED = ['Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'shih-tzu',
         'Blenheim_spaniel', 'papillon', 'toy_terrier', 'Rhodesian_ridgeback',
         'Afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick', 'black-and-tan_coonhound',
         'Walker_hound', 'English_foxhound', 'redbone', 'borzoi', 'Irish_wolfhound',
         'Italian_greyhound', 'whippet', 'Ibizan_hound', 'Norwegian_elkhound',
         'otterhound', 'Saluki', 'Scottish_deerhound', 'Weimaraner',
         'Staffordshire_bullterrier', 'American_Staffordshire_terrier',
         'Bedlington_terrier', 'Border_terrier', 'Kerry_blue_terrier',
         'Irish_terrier', 'Norfolk_terrier', 'Norwich_terrier',
         'Yorkshire_terrier', 'wire-haired_fox_terrier', 'Lakeland_terrier',
         'Sealyham_terrier', 'Airedale', 'cairn', 'Australian_terrier',
         'Dandie_Dinmont', 'Boston_bull', 'miniature_schnauzer',
         'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier',
         'Tibetan_terrier', 'silky_terrier', 'soft-coated_wheaten_terrier',
         'West_Highland_white_terrier', 'Lhasa', 'flat-coated_retriever', 'curly-coated_retriever',
         'golden_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever',
         'german_short-haired_pointer', 'vizsla', 'English_setter', 'Irish_setter',
         'Gordon_setter', 'Brittany_spaniel', 'clumber', 'English_springer',
         'Welsh_springer_spaniel', 'cocker_spaniel', 'Sussex_spaniel',
         'Irish_water_spaniel', 'kuvasz', 'schipperke', 'groenendael',
         'malinois', 'briard', 'kelpie', 'komondor', 'Old_English_sheepdog',
         'Shetland_sheepdog', 'collie', 'Border_collie', 'Bouvier_des_Flandres',
         'Rottweiler', 'German_shepherd', 'Doberman', 'miniature_pinscher',
         'Greater_Swiss_Mountain_dog', 'Bernese_mountain_dog', 'Appenzeller',
         'EntleBucher', 'boxer', 'bull_mastiff', 'Tibetan_mastiff',
         'French_bulldog', 'Great_Dane', 'Saint_Bernard', 'Eskimo_dog',
         'malamute', 'Siberian_husky', 'affenpinscher', 'basenji', 'pug',
         'Leonberg', 'Newfoundland', 'Great_Pyrenees', 'Samoyed', 'Pomeranian',
         'chow', 'keeshond', 'Brabancon_griffon', 'Pembroke', 'Cardigan',
         'toy_poodle', 'miniature_poodle', 'standard_poodle', 'Mexican_hairless',
         'dingo', 'dhole', 'African_hunting_dog']

################## VALIDATIONS #################
env_valid_options = dict(
    MODEL_TARGET=["local", "gcs"],
)
