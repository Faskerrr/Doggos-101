
from tensorflow.keras.utils import image_dataset_from_directory

# Function to load the images for the train, validation and test datasets

def load_train_validation_images(path:str):
    '''
  Load images from the Dog Dataset of Kaggle.
  Returns X and y as numpy arrays
  '''
    data_dir = path
    train_ds = image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=32)
    print("✅ Imported train images, with shape", train_ds.shape)

    validation_ds = image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(224, 224),
        batch_size=32)
    print("✅ Imported validation images, with shape", validation_ds.shape)

    class_names = [item.split("-")[1] for item in train_ds.class_names]
    print("✅ Created the class name, the number of class is ", len(class_names))

    return train_ds, validation_ds, class_names


def load_test_dataset(path:str):
    '''
    Load the test dataset from the Dog Dataset of Kaggle.
    Args:
        path: path to the test dataset
    '''
    test_dir = path

    test_ds = image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='categorical',
        image_size=(224, 224),
        batch_size=32)
    print("✅ Imported test images, with shape", test_ds.shape)

    return test_ds
