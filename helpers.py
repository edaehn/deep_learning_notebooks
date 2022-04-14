############################################################
#           Helper functions for Deep Learning experiments
#  Some code snips are adopted from the
#  https://www.udemy.com/course/tensorflow-developer-certificate-machine-learning-zero-to-mastery/
############################################################

############################### Importing required libraries

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import itertools
import random
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
import matplotlib.image as mpimg
import pathlib
import zipfile


############################### Exploring and preparing data

def unzip_file(filename="10_food_classes_10_percent.zip"):
    """
    Unzips a file with a defined full-path filename.
    :param filename: A filename with absolute path
    :return:
    """
    try:
        zip_ref = zipfile.ZipFile(filename)
        zip_ref.extractall()
        zip_ref.close()
    except Exception as error:
        print(str(error))
        return False
    else:
        return True


def walk_directory(directory):
    """
    Walk through a data directory and list the number of files.
    :param directory: Full path to the directory
    :return:
    """
    for dirpath, dirnames, filenames in os.walk(directory):
        print(f"There are {len(dirnames)} directories and '{len(filenames)}'' files in {dirpath}.")


def get_classnames(dataset_train_directory="sample_data/birds/train/"):
  """
  Get the class names based on names of subdirectories found in the directory
  with the training dataset.
  :param dataset_train_directory: Full path to the directory with training dataset
  :return: a sorted list of class names
  """
  data_dir = pathlib.Path(dataset_train_directory)
  class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
  print(class_names)
  return class_names


def get_image_data(dataset_path="sample_data/birds", IMG_SIZE = (224, 224)):
    """
    Get image datasets for training and testing.
    :param dataset_path: full-path to the directory containing /train and /test subdirectories.
    :param IMG_SIZE: defaults to (224, 224) tuple.
    :return: train_data, test_data
    """

    # Defining train and test directories
    train_directory = dataset_path + "/train"
    test_directory = dataset_path + "/test"

    # Setup data inputs
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_directory,
        label_mode="categorical",
        image_size=IMG_SIZE
    )

    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        test_directory,
        label_mode="categorical",
        image_size=IMG_SIZE
    )

    # Return datasets
    return train_data, test_data


def get_normalised_image_data(directory="10_food_classes_10_percent",
             IMAGE_SHAPE = (224, 224),
             BATCH_SIZE = 32,
             class_mode="categorical",
             rescale=1 / 255.):
    """
    With the help of ImageDataGenerator, get_data() returns training and
    test datasets based on an input directory file contents.
    :param directory: Full path to a directory with /train and /test subdirectories
    containing respective datasets
    :param IMAGE_SHAPE: defaults to (224, 224) image size
    :param BATCH_SIZE: number of batches to preprocess
    :param class_mode: defaults to "categorical"
    :param rescale: rescales images from RGB-based values to values [0..1]
    :return: train_data, test_data for training and test datasets
    """

    train_dir = directory + "/train/"
    test_dir = directory + "/test/"

    train_datagen = ImageDataGenerator(rescale=rescale)
    test_datagen = ImageDataGenerator(rescale=rescale)

    print("Training images:")
    train_data = train_datagen.flow_from_directory(train_dir,
                                                              target_size=IMAGE_SHAPE,
                                                              batch_size=BATCH_SIZE,
                                                              class_mode=class_mode)

    print("Testing images:")
    test_data = test_datagen.flow_from_directory(test_dir,
                                                            target_size=IMAGE_SHAPE,
                                                            batch_size=BATCH_SIZE,
                                                            class_mode=class_mode)
    return train_data, test_data


def preprocess_and_augment_data(directory="sample_data/birds"):
    """
    Creating augmented training dataset and test dataset based on the file
    contents of an input directory. Both datasets are normalised.
    The training dataset is augmented with rotation, zooming, horisontal flips,
    width and height shifts.
    :param directory: A full-path to the directory with images
    :return: train_data_augmented, test_data for training (augmented)
    and test datasets
    """

    train_dir = directory + "/train/"
    test_dir = directory + "/test/"

    # Create ImageDataGenerator training instance with data augmentation
    train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=0.2,

                                             zoom_range=0.2,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             horizontal_flip=True)

    train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                             target_size=(224, 224),
                                             batch_size=32,
                                             class_mode="categorical",
                                             shuffle=True)
    # Rescale (normalisation)
    test_datagen = ImageDataGenerator(rescale=1/255.)

    test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=(224, 224),
                                             batch_size=32,
                                             class_mode="categorical",
                                             shuffle=True)
    return train_data_augmented, test_data


############################### Plotting performance curves

def plot_loss_curves(history, metric="accuracy"):
    """
    Plot loss and performance (defaults to accuracy) curves for
    training and validation history
    :param history: History of model fitting
    :param metric: defaults to accuracy
    :return: True if the metric data is in the history object, otherwise False
    """
    if not(metric in history.history.keys()):
        return False

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    accuracy = history.history[metric]
    val_accuracy = history.history["val_"+metric]

    epochs = range(len(history.history["loss"]))

    # Plot loss
    plt.plot(epochs, loss, label="Training loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.legend()

    # Plot the accuracy or other metric
    plt.figure();
    plt.plot(epochs, accuracy, label="Training "+metric)
    plt.plot(epochs, val_accuracy, label="Validation "+metric)

    plt.title(metric)
    plt.xlabel("epochs")
    plt.legend()
    return True


def compare_histories(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow History objects.
    :param original_history: original history object;
    :param new_history: new history object;
    :param initial_epochs: number of epochs in the original history;
    :return: 
    """
    # Get original history measurements
    accuracy = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    # Validation accuracy and loss
    val_accuracy = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history metrics with new history metrics
    total_accuracy = accuracy + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    # Combine original history with new history metrics for validation tests
    total_val_accuracy = val_accuracy + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Draw plots for accuracy
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_accuracy, label="Training Accuracy")
    plt.plot(total_val_accuracy, label="Validation Accuracy")

    # Plot a line where the fine-tuning started
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label="Start Fine-tuning")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    # Draw plots for loss
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label="Training Loss")
    plt.plot(total_val_loss, label="Validation Loss")

    # Plot a line where the fine-tuning started
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label="Start Fine-tuning")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")


############################### Visualising  images

def view_random_image(target_dir="sample_data/birds/train/", \
                      target_class="ALPINE CHOUGH"):
    """
    Show and return a random image from target directory and class subdirectory.
    :param target_dir: a directory with a training dataset
    :param target_class: a subdirectory inside of the training dataset
    :return: False if the dircetory does not exists, ortherwise returns image
    """

    # Setup the target directory
    target_folder = target_dir + target_class
    print(target_folder)
    if not os.path.isdir(target_folder):
       return False

    # Get a random image path
    random_image = random.sample(os.listdir(target_folder), 1)
    print(random_image)

    # Read and plot the image
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off");

    # Show the image shape
    print(f"Image shape: {img.shape}")

    # Return the image
    return img


def show_five_birds(dataset_path="sample_data/birds"):
    """
    Shows five random training images (birds) from a directory (should contain '/train' subdirectory) with training and
    test datasets
    :param dataset_path: Full-path to images dataset
    :return:
    """

    # Select five random images from five random categories (respective to
    # the subdirectory names)
    image_filenames = []
    target_classes = random.choices(os.listdir(dataset_path + "/train/"), k=5)

    plt.figure(figsize=(20, 4))
    count = 1
    for target_class in target_classes:
        plt.subplot(1, 5, count)
        bird_img = view_random_image(dataset_path + "/train/", target_class);
        count +=1


# Prepare an image for prediction
def load_and_prepare_image(filename, img_shape=224, rescale=True):
    """
    Preparing an image for image prediction task.
    Reads and reshapes the tensor into needed shape.
    Image tensor is rescaled.
    :param filename: full-path filename of the image
    :param img_shape: required shape of the output image
    :param rescale: is True when we return normalised image tensor
    :return: image tensor
    """

    # Read the image
    img = tf.io.read_file(filename)

     # Decode the image into tensorflow
    img = tf.image.decode_image(img)

    # Resize the image
    img = tf.image.resize(img, size = [img_shape, img_shape])

    # Rescale the image
    if rescale:
        img = img/255.

    return img

def predict_and_plot(model, filename, class_names, known_label=False, rescale=True):
    """
    Loads an image stored at filename, makes the prediction,
    plots the image with the predicted class as the title.
    :param model:  Multi-class/Binary classification model.
    :param filename: filename of the image to predict.
    :param class_names: class names of the model.
    :param known_label: if we want to compare the known
    label with the predicted label.
    :return:
    """

    # import the target image and preprocess it
    img = load_and_prepare_image(filename, rescale=rescale)

    # Make a prediction
    predicted = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    # Check for multi-class classification
    # print(predicted)
    if len(predicted[0])>1:
      predicted_class = class_names[tf.argmax(predicted[0])]
    else:
      # Binary classification
      predicted_class = class_names[int(tf.round(predicted[0]))]

    # Plot the image and predicted class
    plt.figure(figsize=(5,5))
    if rescale:
      plt.imshow(img);
    else:
      plt.imshow(img/255.);

    if known_label:
        if (known_label == predicted_class):
            plt.title(f"Predicted correctly: {predicted_class}")
        else:
            plt.title(f"{known_label } predicted as {predicted_class}")
    else:
        plt.title(f"Predicted: {predicted_class}")
    plt.axis(False)


############################### Callbacks

def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + \
            datetime.datetime.now().strftime("%Y%m%d-%H")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

# Create a model check-point callback, only saving the model weights
def create_checkpoint_callback(checkpoint_path=\
          "tmp/ten_percent_checkpoints_weights/checkpoint.ckpt"):
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                           save_weights_only=True,
                                                           save_freq="epoch",
                                                           save_best_only=False,
                                                           verbose=1)

  return checkpoint_callback

############################### Model Creation

def create_model(model_url, num_classes=10, IMAGE_SHAPE=(224, 224)):
    """
    Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.

    Args:
      model_url(str): A TensorFlow Hub feature extraction URL.
      num_classes(int): Number of output neurons in the output layer,
      should be equal to number of target classes, default 10.
      IMAGE_SHAPE: shape of images.

    Returns:
      An uncompiled Keras Sequential model with model_url as feature extractor
      layer and Dense output layer with num_classes output neurons.
    """
    # Download the pretrained model and save it as a Keras layer
    feature_extraction_layer = hub.KerasLayer(model_url,
                                              trainable=False,  # Freeze the alteday learned patterns
                                              name="feature_extraction_layer",
                                              input_shape=IMAGE_SHAPE + (3,))

    # Create our own model
    model = tf.keras.Sequential([
        feature_extraction_layer,
        layers.Dense(num_classes, activation="softmax",
                     name="output_layer")
    ])

    # Compile our model
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    return model