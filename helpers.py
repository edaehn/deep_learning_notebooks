############################################################
#           Helper functions for Deep Learning experiments
############################################################

# Importing required libraries
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

############################### Exploring and preparing data

def walk_directory(directory):
    # Walk through a data directory and list number of files
    for dirpath, dirnames, filenames in os.walk(directory):
        print(f"There are {len(dirnames)} directories and '{len(filenames)}'' files in {dirpath}.")

# Get the classnames programatically
def get_classnames(dataset_train_directory="sample_data/birds/train/"):
  # Get the classnames programatically
  data_dir = pathlib.Path(dataset_train_directory)
  class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
  print(class_names)
  return class_names
# class_names = get_classnames()

def get_data(directory="10_food_classes_10_percent"):
    # HYPERPARAMETERS
    IMAGE_SHAPE = (224, 224)
    BATCH_SIZE = 32

    train_dir = directory + "/train/"
    test_dir = directory + "/test/"

    train_datagen = ImageDataGenerator(rescale=1 / 255.)
    test_datagen = ImageDataGenerator(rescale=1 / 255.)

    print("Training images:")
    train_data = train_datagen.flow_from_directory(train_dir,
                                                              target_size=IMAGE_SHAPE,
                                                              batch_size=BATCH_SIZE,
                                                              class_mode="categorical")

    print("Testing images:")
    test_data = test_datagen.flow_from_directory(test_dir,
                                                            target_size=IMAGE_SHAPE,
                                                            batch_size=BATCH_SIZE,
                                                            class_mode="categorical")
    return train_data, test_data

# Normalise training and testing data.
# Augment the training data
def preprocess_and_augment_data(train_dir, test_dir):
  # Create ImageDataGenerator training instance with data augmentation
  train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=0.2,

                                             zoom_range=0.2,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             horizontal_flip=True)

  train_data_augmented = train_datagen_augmented.flow_from_directory("sample_data/birds/train/",
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

train_data_augmented, test_data = preprocess_and_augment_data(train_dir="sample_data/birds/train/",
                                        test_dir="sample_data/birds/test/")

def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation matrix
  """
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]

  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]

  epochs = range(len(history.history["loss"]))

  # Plot loss
  plt.plot(epochs, loss, label="Training loss")
  plt.plot(epochs, val_loss, label="Validation loss")
  plt.title("Loss")
  plt.xlabel("epochs")
  plt.legend()

  # Plot the accuracy
  plt.figure();
  plt.plot(epochs, accuracy, label="Training accuracy")
  plt.plot(epochs, val_accuracy, label="Validation accuracy")
  plt.title("Accuracy")
  plt.xlabel("epochs")
  plt.legend()

# Visualise our images
def view_random_image(target_dir, target_class):
  # Setup the target directory
  target_folder = target_dir + target_class

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

  return img
# img = view_random_image(target_dir="sample_data/birds/train/", target_class="ALPINE CHOUGH")

def show_five_birds():
    plt.figure(figsize=(20, 4))
    plt.subplot(1, 5, 1)
    bird_img = view_random_image("sample_data/birds/train/", "SHOEBILL")
    plt.subplot(1, 5, 2)
    bird_img = view_random_image("sample_data/birds/train/", "RED BEARDED BEE EATER")
    plt.subplot(1, 5, 3)
    bird_img = view_random_image("sample_data/birds/train/", "POMARINE JAEGER")
    plt.subplot(1, 5, 4)
    bird_img = view_random_image("sample_data/birds/train/", "WATTLED CURASSOW")
    plt.subplot(1, 5, 5)
    bird_img = view_random_image("sample_data/birds/train/", "STORK BILLED KINGFISHER")

# Prepare an image for prediction
def load_and_prepare_image(filename, img_shape=224):
  """
  Preparing an image for image prediction task.
  Reads and reshapes the tensor into needed shape.
  """

  # Read the image
  img = tf.io.read_file(filename)

  # Decode the image into tensorflow
  img = tf.image.decode_image(img)

  # Resize the image
  img = tf.image.resize(img, size = [img_shape, img_shape])

  # Rescale the image
  img = img/255.

  return img

def predict_and_plot(model, filename, class_names, known_label=False):
    """
    Imports an image at filename, makes the prediction,
    plots the image with the predicted class as the title.
    """

    # import the target image and preprocess it
    img = load_and_prepare_image(filename)

    # Make a prediction
    predicted = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    # Check for multi-class classification
    print(predicted)
    if len(predicted[0])>1:
      predicted_class = class_names[tf.argmax(predicted[0])]
    else:
      # Binary classification
      predicted_class = class_names[int(tf.round(predicted[0]))]

    # Plot the image and predicted class
    plt.figure(figsize=(5,5))
    plt.imshow(img)
    if known_label:
        if (known_label == predicted_class):
            plt.title(f"Predicted correctly: {predicted_class}")
        else:
            plt.title(f"{known_label } predicted as {predicted_class}")
    else:
        plt.title(f"Predicted: {predicted_class}")
    plt.axis(False)
# predict_and_plot(baseline_model,
#                  filename="/content/sample_data/birds/test/ALPINE CHOUGH/2.jpg",
#                  class_names=class_names, known_label="ALPINE CHOUGH")

############################### Callbacks

def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + \
            datetime.datetime.now().strftime("%Y%m%d-%H")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

############################### Model Creation

# Create models from a URL
def create_model(model_url, num_classes=10):
    """
    Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.

    Args:
      model_url(str): A TensorFlow Hub feature extraction URL.
      num_classes(int): Number of output neurons in the output layer,
      should be equal to number of target classes, default 10.

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