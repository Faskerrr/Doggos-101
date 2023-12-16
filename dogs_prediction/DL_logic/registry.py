# registry.py is for loading and storing the model's weights from/to Cloud Storage
import os
from colorama import Fore, Style
import tensorflow as tf
from dogs_prediction.params import *
from dogs_prediction.DL_logic.utils import LayerScale

def load_convnext_model(model_name='ConvNeXtXLargeNew'):
    '''
    Loads ConvNeXtXLarge model with accuracy ~92% from local storage
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, '../..', 'models'))

    print(Fore.BLUE + f"\nLoad model '{model_name}' from local registry..." + Style.RESET_ALL)
    model_path = os.path.join(models_dir, f"{model_name}.keras")
    if not os.path.exists(model_path):
        print(f"\n❌ No model with name '{model_name}' found in local models directory")
        return None
    else:
        model = tf.keras.models.load_model(model_path, compile=False, custom_objects={"LayerScale": LayerScale})
        print("✅ Model loaded from local")
        return model
