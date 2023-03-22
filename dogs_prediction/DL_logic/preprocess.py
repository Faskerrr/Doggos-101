from tensorflow.keras.applications import vgg19, resnet50, inception_v3, mobilenet_v2, xception
# Function to preprocess the images for the train, validation and test datasets

def preprocess(img, label, model_name):
    img = model_name.preprocess_input(img)
    return img, label
