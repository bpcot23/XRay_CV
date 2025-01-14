import numpy as np
import pandas as pd
import re 
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class FileProcessor:

    def __init__(self, config):
        self.config = config
        self.dataset = config["file_processing"]["dataset"]

        self.augmentation = config["file_processing"]["augmentation"]
        self.augment_with_smote = config["file_processing"]["augment_with_SMOTE"]
        self.num_aug_per_image = config["file_processing"]["num_aug_per_image"]
        self.undersample_majority = config["file_processing"]["undersample_majority"]

        self.conditions = config["file_processing"]["multi_condition_dataset"]["condition_list"]
        self.target_condition = config["file_processing"]["target_condition"]
        self.image_height = config["file_processing"]["image_height"]
        self.image_width = config["file_processing"]["image_width"]
        self.image_path = config["file_processing"]["multi_condition_dataset"]["image_directory"]
        self.batch_size = config["file_processing"]["batch_size"]
        self.image_key = None


    def create_key(self):
        """
        Initialize a pandas dataframe containing metadata on each image and its condition classifications.

        Args:
            None

        Returns:
            image_key (pd.DataFrame): A dataframe containing gold labels for each image, patient, image dimensions, and pixel spacing
        """
        image_key = pd.read_csv(self.config["file_processing"]["key_filepath"],
                                sep = ",",
                                encoding = "utf-8-sig")
        
        # Create columns for each condition
        label_index = 2
        for condition in self.conditions:
            image_key.insert(label_index, condition, "", True)
            label_index += 1

        def assign_labels(row):
            """
            Creates new rows for each condition, and assigns labels of present/absent based on the dataset's gold labels,

            Args:
                row (pd.DataFrame): One row of the dataframe

            Returns:
                row (pd.DataFrame): Updated row of the dataframe, now with binary condition labels
            """
            present_conditions_raw = row['Finding Labels']
            present_conditions = present_conditions_raw.split("|")
            present_conditions = list(map(lambda x: x.strip().lower(), present_conditions))

            for condition in self.conditions:
                if condition in present_conditions:
                    row[condition] = 1
                else:
                    row[condition] = 0
            return row

        # Assign binary present/absent label for each condition to each image
        image_key = image_key.apply(assign_labels, axis = 1)

        self.image_key = image_key
        return image_key
    
    
    def dataset_split(self):
        """
        Splits the dataset into train and test datasets for further model processing.

        Args:
            None
        
        Returns:
            train_key (pd.DataFrame): Dataframe containing all metadata for the training image set
            test_key (pd.DataFrame): Dataframe containing all metadata for the test image set
        """
        if self.dataset == "pneumonia":
            image_key = pd.read_csv(self.config["file_processing"]["key_filepath"],
                        sep = ",",
                        encoding = "utf-8-sig")
            
            # Divide image_key into a set for pneumonia cases and a set for normal cases
            pneumonia_key = image_key[image_key["pneumonia"] == 1]
            
            # If dataset balancing occurs through majority undersampling
            if self.undersample_majority == "True":
                pneumonia_key, undersampled_leftovers = train_test_split(pneumonia_key, test_size = 0.4, random_state = 5)

            p_train_key, p_test_key = train_test_split(pneumonia_key, test_size = 0.3, random_state = 9)
            p_validation_key, p_test_key = train_test_split(p_test_key, test_size = 0.5, random_state = 42)

            normal_key = image_key[image_key["pneumonia"] == 0]
            n_train_key, n_test_key = train_test_split(normal_key, test_size = 0.3, random_state = 9)
            n_validation_key, n_test_key = train_test_split(n_test_key, test_size = 0.5, random_state = 42)

            # Rejoin the two datasets to ensure uniform normal/abnormal ratio across training, validation, and testing
            train_key = pd.concat([p_train_key, n_train_key])
            validation_key = pd.concat([p_validation_key, n_validation_key])
            test_key = pd.concat([p_test_key, n_test_key])

        else:    
            image_key = self.image_key
            train_key, test_key = train_test_split(image_key, test_size = 0.3, random_state = 9)
            validation_key, test_key = train_test_split(test_key, test_size = 0.5, random_state = 42)

        # For dataset visualization, comment in or out as desired
        # print("train key:")
        # print(train_key)
        # train_key.to_csv("train_key.csv")
        # print("validation key:")
        # print(validation_key)
        # validation_key.to_csv("validation_key.csv")
        # print("test key:")
        # print(test_key)
        # test_key.to_csv("test_key.csv")

        # Calls for data augmentation via image shifting or SMOTE
        if self.augmentation == "True":
            if self.augment_with_smote == "True":
                print("generating augmented dataset through SMOTE")
                train_dataset = self.load_and_smote(train_key)
            else:
                print("generating augmented image set through image transformations")
                train_dataset = self.augment_dataset(train_key)
        else:
            train_dataset = self.load_image_sets(train_key) 

        validation_dataset = self.load_image_sets(validation_key)
        test_dataset = self.load_image_sets(test_key)  

        return train_dataset, validation_dataset, test_dataset
    

    def augment_dataset(self, train_key):
        """
        Takes in an imbalanced dataset and generates augmented data to balance the classes through image transformations.
        Then generates a tensorflow dataset through image preprocessing.

        Args:
            train_key (pd.DataFrame): a dataframe containing metadata on each image in the dataset

        Returns:
            dataset (tf.data.Dataset): a dataset of images and condition labels, augmented to include balanced classes
        """
        if self.dataset == "pneumonia":
            # Separate minority (condition absent) and majority (condition present) classes
            minority_class = train_key[train_key[self.target_condition] == 0]
            majority_class = train_key[train_key[self.target_condition] == 1]            

        else:
            # Separate minority (condition present) and majority (condition absent) classes
            minority_class = train_key[train_key[self.target_condition] == 1]
            majority_class = train_key[train_key[self.target_condition] == 0]

        augmented_image_set = []
        augmented_label_set = []
        majority_images = []
        majority_labels = []

        data_generator = ImageDataGenerator(
                                rotation_range=20,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                shear_range=0.1,
                                zoom_range=0.1,
                                horizontal_flip=False,
                                fill_mode='nearest'
                                )
        
        # Preprocess minority class images and generate augmented images
        for _, row in minority_class.iterrows():

            if self.dataset == "pneumonia":
                image_path = row["Image Index"]
            else:
                image_path = os.path.join(self.image_path, row["Image Index"])

            label = row[self.target_condition]

            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)

            # Convert to RGB: dataset in greyscale, model expects RGB
            if tf.shape(image)[-1] == 1:
                image = tf.image.grayscale_to_rgb(image)
            image = tf.image.resize(image, [self.image_height, self.image_width])
            image = tf.keras.applications.efficientnet.preprocess_input(image)
            image = tf.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))

            for _ in range(self.num_aug_per_image):
                augmented_image = next(data_generator.flow(image, batch_size=1))[0]
                augmented_image_set.append(augmented_image)

                # Inherits the image label of the pre-augmented image to keep them grouped together
                augmented_label_set.append(label)

        # Preprocess majority class images
        for _, row in majority_class.iterrows():

            if self.dataset == "pneumonia":
                image_path = row["Image Index"]
            else:
                image_path = os.path.join(self.image_path, row["Image Index"])


            label = row[self.target_condition]
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [self.image_height, self.image_width])
            image = tf.keras.applications.efficientnet.preprocess_input(image)
            majority_images.append(image)
            majority_labels.append(label)

        augmented_image_set = np.array(augmented_image_set)
        augmented_label_set = np.array(augmented_label_set)
        majority_images = np.array(majority_images)
        majority_labels = np.array(majority_labels)

        # Combine all data
        all_images = np.concatenate([majority_images, augmented_image_set], axis=0)
        all_labels = np.concatenate([majority_labels, augmented_label_set], axis=0)

        # Create dataset using TensorFlow
        dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
        dataset = dataset.shuffle(buffer_size=len(all_labels)).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset


    def load_image_sets(self, image_set):
        """
        Loads in images into a dataset, standardizes their dimensions through preprocessing.

        Args:
            image_set (pd.DataFrame): a dataframe containing metadata on a subset of the full dataset

        Returns:
            dataset (tf.data.Dataset): a dataset of images and condition labels

        """
        if self.dataset == "pneumonia":
            image_paths = image_set["Image Index"].values
        else:
            image_set['image_path'] = image_set['Image Index'].apply(lambda x: os.path.join(self.image_path, x))
            image_paths = image_set['image_path'].values

        labels = image_set[self.target_condition].values
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        # Preprocess the image set and shuffle the resulting dataset
        dataset = dataset.map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=len(image_set)).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
    
    
    def load_and_smote(self, image_set):
        """
        Loads images into a dataset, applies SMOTE to balance classes, and standardizes image dimensions.

        Args:
            image_set (pd.DataFrame): Metadata containing image paths and labels.

        Returns:
            dataset (tf.data.Dataset): TensorFlow dataset with balanced classes.
        """
        if self.dataset == "pneumonia":
            image_paths = image_set["Image Index"].values
        else:
            image_set['image_path'] = image_set['Image Index'].apply(lambda x: os.path.join(self.image_path, x))
            image_paths = image_set['image_path'].values

        labels = image_set[self.target_condition].values

        # Apply preprocessing
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

        # Convert dataset to NumPy arrays for SMOTE
        images = []
        labels = []
        # Flatten images for SMOTE, create arrays for images and labels
        for image, label in dataset:
            images.append(image.numpy().flatten())  
            labels.append(label.numpy())
        images = np.array(images)
        labels = np.array(labels)

        # Apply SMOTE
        smote = SMOTE(random_state=9)
        images_resampled, labels_resampled = smote.fit_resample(images, labels)

        # Reshape images back to original dimensions
        images_resampled = images_resampled.reshape((-1, self.image_height, self.image_width, 3))

        # Create TensorFlow dataset from balanced data
        dataset_balanced = tf.data.Dataset.from_tensor_slices((images_resampled, labels_resampled))
        dataset_balanced = dataset_balanced.shuffle(buffer_size=len(labels_resampled)).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset_balanced
    

    def preprocess_image(self, image_path, label):
        """
        Resize and preprocess an image to standardize images in dataset creation.

        Args:
            image_path (str): file path to the given image within the data directory
            label (int): a 1 if the condition is present, a 0 if the condition is absent
        """
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.image_height, self.image_width])
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        return image, label