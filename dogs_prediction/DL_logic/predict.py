import tensorflow as tf
import numpy as np
from dogs_prediction.params import *
from dogs_prediction.DL_logic.utils import *

def predict_labels(model, *args, **kwargs):
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
    img = tf.keras.preprocessing.image.img_to_array(img)    #shape = (224, 224, 3)
    img = img.reshape((-1, 224, 224, 3))
    print("✅ Image successfully reshaped", img.shape)

    model = compile_model(model)
    print("✅ Model successfully loaded and compiled")

    res = model.predict(img)
    print("✅ Breed predicted")
    indexes = np.argsort(res)[0][-3:][::-1]
    predicts = np.sort(res)[0][::-1][0:3]

    breed_prediction = {
        'first': BREED[indexes[0]],
        'second': BREED[indexes[1]],
        'third': BREED[indexes[2]],
    }
    score_prediction = {
        'first': float(round(predicts[0],8)),
        'second': float(round(predicts[1],8)),
        'third': float(round(predicts[2],8))
    }
    output = {'prediction': breed_prediction,
              'score': score_prediction}

    return output
