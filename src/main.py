import json
import os
import re

from file_processor import FileProcessor

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
    
def main(config):
    """
    Main function for running the XRay classification system using EfficientNet

    Args:
        config: json path for fetching all global variable values and filepaths.
    """
    # Initialize FileProcessor class, create training and test datasets
    myFP = FileProcessor(config)
    image_key = myFP.create_key()
    train_key, test_key = myFP.train_test_split()

if __name__ == "__main__":
    config = load_config(path='config.json')
    main(config)