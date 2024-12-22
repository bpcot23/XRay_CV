import pandas as pd
import re 

class FileProcessor:

    def __init__(self, config):
        self.config = config
        self.conditions = config["file_processing"]["condition_list"]
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
            present_conditions = present_conditions_raw.split("|").strip().lowercase()

            for condition in self.conditions:
                if condition in present_conditions:
                    row[condition] = "present"
                else:
                    row[condition] = "absent"
            return row

        # Assign binary present/absent label for each condition to each image
        image_key = image_key.apply(assign_labels, axis = 1)

        self.image_key = image_key
        return image_key
    
    
    def train_test_split(self):
        """
        Splits the dataset into train and test datasets for further model processing.

        Args:
            None
        
        Returns:
            train_key (pd.DataFrame): Dataframe containing all metadata for the training image set
            test_key (pd.DataFrame): Dataframe containing all metadata for the test image set
        """
        image_key = self.image_key
        image_key.insert(0, "group", "", True)

        # Read in file for train/test split
        testlist_filepath = self.config["file_processing"]["test_list"]
        with open(testlist_filepath, "r") as testlist:
            test_images = testlist.readlines()

        def assign_train_test_label(self, row):
            """
            Assigns group value of "test" to all files listed in test_images, otherwise assigns "train"

            Args:
                row (pd.DataFrame): One row of the dataframe
            
            Returns:
                row (pd.DataFrame): Updated row of the dataframe, now with a group label
            """
            filename = row["Image Index"]
            if filename in test_images:
                row["group"] = "test"
            else:
                row["group"] = "train"
            return row
        
        image_key = image_key.apply(assign_train_test_label, axis = 1)

        # Initialize train and test datasets, splitting image_key by the "group" column
        train_key = image_key[image_key["group"] == "train"]
        test_key = image_key[image_key["group"] == "test"]

        return train_key, test_key