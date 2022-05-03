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
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import itertools
import random
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomHeight, RandomWidth
import os
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


############################### Plotting confusion matrices

def draw_confusion_matrix(y_test, y_preds, classes=None, figsize = (10, 10), text_size=16, save_fig=False):
    """
    Creates a confusion matrix figure with y_test, y_preds labels
    :param y_test: test labels.
    :param y_preds: predicted labels.
    :param classes: class names.
    :param figsize: size of the confusion matrix figure.
    :param text_size: text size for label text.
    :return:
    """

    # Create the confusion matrix
    cm = confusion_matrix(y_test, tf.round(y_preds))

    # Normalise the confusion matrix
    cm_normalised = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    number_of_classes = cm.shape[0]

    # Draw the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create a matrix plot
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Set labels to classes
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
       xlabel="Predicted Label",
       ylabel="True Label",
       xticks=np.arange(number_of_classes),
       yticks=np.arange(number_of_classes),
       xticklabels=labels,
       yticklabels=labels
       )

    # Set x-axis labels to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Draw x-labels vertically
    plt.xticks(rotation=70, fontsize=text_size)
    plt.yticks(fontsize=text_size)

    # Adjust label size
    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size+4)

    # Set threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_normalised[i, j]*100:.1f})",
           horizontalalignment="center",
           color="white" if cm[i, j] > threshold else "black",
           size=text_size/2)

    # Saving the figure to the current directory
    if save_fig:
        fig.savefig("confusion_matrix.png")


############################### Plotting f1 bar-chart

# Placing values on the bar chart
# a related example for vertical barcharts
# at: https://matplotlib.org/2.0.2/examples/api/barchart_demo.html
def autolabel(rects, ax):
    """
    Attach a text label in center of each horisonal bar
    :param rects: a BarContainer object with artists for each class.
    :param ax: axis of the figure plot.
    """
    for rect in rects:
        width = rect.get_width()
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2.,
                '%.2f' % width,
                ha='center', va='center', color='black')


def plot_f1_barh(model,
                 test_dataset_directory="sample_data/birds/test",
                 figsize=(12, 80)):
    """
    Calculates test accuracy and plots F1-score bar-chart using trained model and
    data from test_directory.
    :param model: trained model.
    :param test_dataset_directory: full path to the test dataset.
    :param figsize: size of the figure with bar-chart.
    :return:
    """

    # Getting test data unshaffled, otherwise we will not a good accuracy score
    # because of the data order
    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        directory=test_dataset_directory,
        label_mode="categorical",
        image_size=(224, 224),
        shuffle=False
    )

    # Use the model for predictions
    prediction_probablities = model.predict(test_data, verbose=1)

    # Getting indices of the predicted classes
    prediction_classes_index = prediction_probablities.argmax(axis=1)

    # Get indices of our test_data BatchDataset
    test_labels = []
    for images, labels in test_data.unbatch():
        test_labels.append(labels.numpy().argmax())

    # Using scikit-learn's accuracy score
    sklearn_accuracy = accuracy_score(y_true=test_labels,
                                      y_pred=prediction_classes_index)
    print(f"Prediction Accuracy on Test Data: {sklearn_accuracy}")

    # Get a dictionary of the classification report
    classification_report_dictionary = classification_report(y_true=test_labels,
                                                             y_pred=prediction_classes_index, output_dict=True)

    # Plot all F1-scores
    f1_scores = {}
    for key, value in classification_report_dictionary.items():
        if key == "accuracy":
            break
        else:
            f1_scores[test_data.class_names[int(key)]] = value["f1-score"]

    # Sort values in the ascending order
    f1_scores_df = pd.DataFrame({"class_names": list(f1_scores.keys()),
                                 "f1-score": list(f1_scores.values())}).sort_values("f1-score", ascending=False)

    # Plotting bar-chart
    fig, ax = plt.subplots(figsize=figsize)
    scores = ax.barh(range(len(f1_scores_df)), f1_scores_df["f1-score"].values)
    autolabel(scores, ax)
    ax.set_yticks(range(len(f1_scores_df)))
    ax.set_yticklabels(f1_scores_df["class_names"])
    ax.set_xlabel("F1-score")
    ax.set_title("F1-scores for 400 Bird Species")
    ax.invert_yaxis()

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
    :param filename (str): full-path filename of the image
    :param img_shape (int): required shape (height/width) of the output image
    :param rescale (bool): is True when we return normalised image tensor
    :return: image tensor
    """

    # Read the image
    img = tf.io.read_file(filename)

     # Decode the image into tensorflow
    img = tf.image.decode_image(img, channels=3)

    # Resize the image
    img = tf.image.resize(img, size = [img_shape, img_shape])

    # Rescale/normalise  the image
    if rescale:
        img = img/255.

    return img

def show_one_wrongly_predicted(filename1, filename2, label1, label2):
    """
    Loads two images from their full-path filenames and show them in one plot with own titles corresponding to their
    class labels.
    :param filename1: full-path filename to the first image, the test image we are predicting.
    :param filename2: full-path to the second image relating to the predicted class.
    :param label1: true class label
    :param label2: predicted class label.
    :return:
    """
    img1 = tf.io.read_file(filename1)
    img1 = tf.image.decode_image(img1, channels=3)
    img2 = tf.io.read_file(filename2)
    img2 = tf.image.decode_image(img2, channels=3)
    figure, ax = plt.subplots(1, 2);
    ax.ravel()[0].imshow(img1);
    ax.ravel()[0].set_title(label1);
    ax.ravel()[0].set_axis_off();
    ax.ravel()[1].imshow(img2);
    ax.ravel()[1].set_title(label2);
    ax.ravel()[1].set_axis_off();
    plt.axis(False);

def show_wrongly_predicted_images(model, dataset_directory="sample_data/birds", top_wrong_predictions_number_to_show=False):

    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        directory=dataset_directory + "/test",
        label_mode="categorical",
        image_size=(224, 224),
        shuffle=False
    )

    class_names = test_data.class_names

    # 2. Use model for predictions
    prediction_probabilities = model.predict(test_data, verbose=1)

    # Check the predictions we have got
    # print(f"Number of test rows: {len(test_data)}, \
            # number of predictions: {len(prediction_probabilities)}, \
            # shape of predcitions: {prediction_probabilities.shape}, \
            # the first prediction: {prediction_probabilities[0]}")

    # Getting indices of the predicted classes
    prediction_classes_index = prediction_probabilities.argmax(axis=1)

    # Get indices of our test_data BatchDataset
    test_labels = []
    for images, labels in test_data.unbatch():
        test_labels.append(labels.numpy().argmax())

    sklearn_accuracy = accuracy_score(y_true=test_labels,
                                      y_pred=prediction_classes_index)

    # 3. Finding where our model is most wrong

    # Find all files in the test dataset
    filepaths = []
    for filepath in test_data.list_files(dataset_directory + "/test/*/*.jpg",
                                         shuffle=False):
        filepaths.append(filepath.numpy())

    # Create a dataframe
    predictions_df = pd.DataFrame({"images_path": filepaths,
                                   "y_true": test_labels,
                                   "y_predicted": prediction_classes_index,
                                   "prediction_confidence": prediction_probabilities.max(axis=1),
                                   "true_classname": [class_names[i] for i in test_labels],
                                   "predicted_classname": [class_names[i] for i in prediction_classes_index]})


    # See which birds predicted correctly/incorrectly
    predictions_df["correct_prediction"] = predictions_df["y_true"] == predictions_df["y_predicted"]

    # Sort out the dataframe to find the most wrongly predicted classes
    top_wrong = predictions_df[predictions_df["correct_prediction"] == False].sort_values("prediction_confidence",
                                                                                              ascending=False)

    # 4. Plot top top_wrong_predictions_number_to_show number of predictions

    top = zip(top_wrong["images_path"], top_wrong["true_classname"], top_wrong["predicted_classname"], top_wrong["prediction_confidence"])
    print(f"Wrongly predicted {len(top_wrong)} out of {len(predictions_df)}")
    if top_wrong_predictions_number_to_show:
        top = top[:top_wrong_predictions_number_to_show]

    for filename1, label1, label2, prob in top:
        filename2 = "/content/sample_data/birds/train/"+ label2 + "/" + random.sample(os.listdir("/content/sample_data/birds/train/" + label2), 1)[0]
        # print(f"{filename1}: {filename2}")
        show_one_wrongly_predicted(filename1, filename2, label1, label2+f" (prob={prob:.2f})")

    return sklearn_accuracy


def predict_image(model, filename, class_names, rescale=True):
    """
    Loads an image stored at filename, makes the prediction.
    :param model:  Multi-class/Binary classification model.
    :param filename: filename of the image to predict.
    :param class_names: class names of the model.
    :param rescale: when True, normalise the image tensor.
    :return: 1. predicted class name and 2. the prediction probabilities, 3. prepared image.
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

    return predicted_class, predicted, img

def predict_and_plot(model, filename, class_names, known_label=False, rescale=True):
    """
    Loads an image stored at filename, makes the prediction,
    plots the image with the predicted class as the title.
    :param model:  Multi-class/Binary classification model.
    :param filename: filename of the image to predict.
    :param class_names: class names of the model.
    :param known_label: if we want to compare the known
    label with the predicted label.
    :param rescale: when True, normalise the image tensor.
    :return:
    """

    predicted_class, predicted, img = predict_image(model=model, filename=filename, class_names=class_names, rescale=rescale)

    # Plot the image and predicted class
    plt.figure(figsize=(5,5))
    if rescale:
      plt.imshow(img);
    else:
      plt.imshow(img/255.);

    if known_label:
        if (known_label == predicted_class):
            plt.title(f"Predicted correctly: {predicted_class} (probability = {predicted.max():.2f})", c="g")
        else:
            plt.title(f"{known_label } predicted as {predicted_class} (probability = {predicted.max():.2f})", c="r")
    else:
        plt.title(f"Predicted: {predicted_class} (probability = {predicted.max():.2f})")
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
          "tmp/ten_percent_checkpoints_weights/checkpoint.ckpt", monitor="val_accuracy"):
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                           save_weights_only=True,
                                                           save_freq="epoch",
                                                           monitor=monitor,
                                                           save_best_only=True,
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
      A compiled Keras Sequential model with model_url as feature extractor
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

############################### Adding data augmentation right into the model

# Create data augmentation stage with horisontal flipping, rotations, zooms, etc.
data_augmentation = Sequential([
                    RandomFlip("horizontal"),
                    RandomRotation(0.2),
                    RandomZoom(0.2),
                    RandomHeight(0.2),
                    RandomWidth(0.2)
                    # Rescaling(1./255) # For models like ResNet50 but not for EffecientNet (having scaling built-in)
], name="data_augmentation")

############################### An EffecientNetB0-based baseline model

def create_baseline_model(input_shape=(224, 224, 3), \
                          number_of_outputs=400, \
                          augment_data=True):
    """
    Builds a headless (no top layers) functional EffecientNetB0 model with
    own output layer. It uses transfer learning feature extraction.

    Args:
      input_shape: shape of images.
      number_of_outputs(int): number of output neurons in the output layer.
      augment_data: when equals to True we do data augmentation.

    Returns:
      A compiled functional feature extraction model with number_of_outputs
      outputs.
    """

    # Setup the baseline model and freeze its layers
    baseline_model = tf.keras.applications.EfficientNetB0(include_top=False)
    baseline_model.trainable = False

    # Create an input layer
    inputs = layers.Input(shape=input_shape, name="input_layer")

    # Add in data augmentation Sequential model as a layer
    if augment_data:
        x = data_augmentation(inputs)  # Uncomment it for data augmentation
        # Give baseline_model the inputs (after augmentation) and don't train it
        x = baseline_model(x, training=False)
    else:
        x = baseline_model(inputs, training=False)

    # Pool output features of the baseline model
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)

    # Put a dense layer on as the output
    outputs = layers.Dense(number_of_outputs, activation="softmax", name="output_layer")(x)

    # Make a model using the inputs and outputs
    model = tf.keras.Model(inputs, outputs)

    # Compile the mopdel
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    # Fit the model
    # history_birds_1 = model.fit(train_data,
    #                              epochs=5,
    #                              steps_per_epoch=len(train_data),
    #                              validation_data=test_data,
    #                              validation_steps=int(0.25*len(test_data)),
    #                              callbacks=[create_tensorboard_callback(dir_name="transfer_learning_birds",
    #                                        experiment_name="birds_baseline_model_1")])

    return model


############################### Fine-tuning the baseline model

def create_tuned_baseline_model(train_data, test_data, \
                                layers_to_unfreeze=5, \
                                input_shape=(224, 224, 3), \
                                number_of_outputs=400, \
                                learning_rate=0.001, \
                                feature_extraction_epochs=5, \
                                augment_data=True):
  """
  Builds a headless (no top layers) functional EffecientNetB0 model with
  own output layer. It uses transfer learning feature extraction.

  Args:
    train_data: training dataset.
    test_data: test dataset.
    layers_to_unfreeze(int): number of top layers to unfreeze for tuning.
    input_shape: shape of images.
    number_of_outputs(int): number of output neurons in the output layer.
    learning_rate: Starting learning rate for the Adam optimiser.
    feature_extraction_epochs(int): The number of epochs the feature extraction
    step is performed. The fine-tuning is done in next feature_extraction_epochs.
    augment_data: when equals to True we do data augmentation.

  Returns:
    model: a compiled and trained functional fine-tuned model with number_of_outputs
    outputs.
    history_feature_extraction: training history of the feature extraction model.
    history_fine_tuning: training history of the fine-tuned model.
  """

  # Setup the baseline model and freeze its layers
  baseline_model = tf.keras.applications.EfficientNetB0(include_top=False)
  baseline_model.trainable = False

  # Create an input layer
  inputs = layers.Input(shape=input_shape, name="input_layer")

  # Add in data augmentation Sequential model as a layer
  if augment_data:
    x = data_augmentation(inputs) # Uncomment it for data augmentation
    # Give baseline_model the inputs (after augmentation) and don't train it
    x = baseline_model(x, training=False)
  else:
    x = baseline_model(inputs, training=False)

  # Pool output features of the baseline model
  x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)

  # Put a dense layer on as the output
  outputs = layers.Dense(number_of_outputs, activation="softmax", name="output_layer")(x)

  # Make a model using the inputs and outputs
  model = tf.keras.Model(inputs, outputs)

  # Compile the mopdel
  model.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=["accuracy"])

  # Fit the model
  print("----- 1. Feature extraction step -----")
  history_feature_extraction = model.fit(train_data,
                                epochs=feature_extraction_epochs,
                                steps_per_epoch=len(train_data),
                                validation_data=test_data,
                                validation_steps=int(0.25*len(test_data)),
                                callbacks=[create_tensorboard_callback(dir_name="transfer_learning_birds",
                                           experiment_name="feature_extraction_baseline"),
                                           create_checkpoint_callback(checkpoint_path=\
                                           "tmp/feature_extraction_birds/checkpoint.ckpt")])
  # Evaluate the feature extraction model
  print("----- 2. Evaluation feature extraction model -----")
  model.evaluate(test_data)

  # Unfreeze all layers in the baseline model
  baseline_model.trainable = True

  # Freeze all layers except of the last layers_to_unfreeze
  for layer in baseline_model.layers[:-layers_to_unfreeze]:
    layer.trainable = False

  # Re-compile the model
  model.compile(loss="categorical_crossentropy",
                # The "rule of thumb" is to recompile the Adam optimiser
                # with the learning rate lesser in 10x than the initial one.
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate/10),
                metrics=["accuracy"])

  # Re-fit the tuned model
  print("----- 3. Fine-tuning step -----")
  history_fine_tuning = model.fit(train_data,
                                epochs=feature_extraction_epochs*2,
                                steps_per_epoch=len(train_data),
                                validation_data=test_data,
                                validation_steps=int(0.25*len(test_data)),
                                initial_epoch=history_feature_extraction.epoch[feature_extraction_epochs-1],
                                callbacks=[create_tensorboard_callback(dir_name="transfer_learning_birds",
                                           experiment_name="fine_tuned_baseline"),
                                           create_checkpoint_callback(checkpoint_path=\
                                           "tmp/fine_tuning_birds/checkpoint.ckpt")])


  # Evaluate the fine-tuned model
  print("----- 4. Evaluate the fine-tuned -----")
  model.evaluate(test_data)

  # Save the fine-tuned model
  # model.save("birds_baseline_fine_tuned")

  # To load the saved model
  # loaded_model = tf.keras.models.load_model("birds_baseline_fine_tuned")

  return model, history_feature_extraction, history_fine_tuning