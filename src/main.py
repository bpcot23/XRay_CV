import json
import numpy as np
import pandas as pd
import os
import re
import tensorflow as tf

from file_processor import FileProcessor
from model import EfficientNet

def load_config(config_path):
    """
    Load config from a given path.

    Args:
        config_path (str): The path to the config file.

    Returns:
        dict: The loaded config if successful, otherwise None.
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def save_dataset(dataset, file_path):
    """
    Saves the augmented dataset after unbatching it for future model runs

    Args:
        dataset (tf.Dataset): A dataset of the images and their gold labels, including images generated through augmentation
        file_path (str): String path to the desired file location of the saved dataset

    Returns:
        None
    """
    images = []
    labels = []

    # Iterate through the dataset (already batched)
    for image_batch, label_batch in dataset:

        # Convert the batched images and labels to numpy arrays
        images.extend(image_batch.numpy())
        labels.extend(label_batch.numpy()) 

    # Convert the list of images and labels to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    np.savez(file_path, images=np.array(images), labels=np.array(labels))


def load_saved_dataset(file_path, batch_size):
    """
    Loads the previously augmented dataset and batches it for use in the model

    Args:
        file_path (str): String path to the desired file location of the saved dataset
        batch_size (int): Size of the data batches for model training

    Returns:
        dataset (tf.Dataset): A dataset of the images and their gold labels, including images generated through augmentation
    """
    # Load the data from the .npz file
    data = np.load(file_path)
    images = data['images']
    labels = data['labels']

    # Convert the data into a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def main(config):
    """
    Main function for running the XRay classification system using EfficientNet

    Args:
        config: json path for fetching all global variable values and filepaths.
    """
    # If saving data, load, split, and augment dataset
    if config["file_processing"]["save_or_load_data"] == "save":

        # Initialize FileProcessor class, create training and test datasets
        print("creating file processor")
        myFP = FileProcessor(config)

        # Necessary step for the multi-condition dataset to process condition labels
        if config["file_processing"]["dataset"] != "pneumonia":
            image_key = myFP.create_key()

        print("splitting train, validation, and test datasets")
        train_dataset, validation_dataset, test_dataset = myFP.dataset_split()

        save_dataset(train_dataset, config["file_processing"]["saved_trainpath"])
        save_dataset(validation_dataset, config["file_processing"]["saved_trainpath"])
        save_dataset(test_dataset, config["file_processing"]["saved_trainpath"])

    
    # Load pre-saved datasets
    else:
        print("Loading datasets from pre-saved files")
        train_dataset = load_saved_dataset(config["file_processing"]["saved_trainpath"], 
                                           config["file_processing"]["batch_size"])
        validation_dataset = load_saved_dataset(config["file_processing"]["saved_valpath"], 
                                                config["file_processing"]["batch_size"])
        test_dataset = load_saved_dataset(config["file_processing"]["saved_testpath"], 
                                          config["file_processing"]["batch_size"])

    # Initialize the EfficientNet class to build, train, and generate predictions from the model
    EffNet = EfficientNet(config)
    print("building the EfficientNet model")
    EffNet.build_model()
    print("training the EfficientNet model")
    model_history = EffNet.train_model(train_dataset = train_dataset, validation_dataset = validation_dataset)
    
    # Optional model fine tuning
    if config["model_building"]["fine_tuning"] == "True":
        print("fine-tuning the EfficientNet model")
        EffNet.unfreeze_layers()
        model_fine_tuned_history = EffNet.train_model(train_dataset = train_dataset, validation_dataset = validation_dataset)

    print("generating predictions using the EfficientNet model")
    model_results = EffNet.generate_predictions(test_dataset = test_dataset)

if __name__ == "__main__":
    config = load_config(config_path= '../config.json')
    main(config)