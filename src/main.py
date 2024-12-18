import json
import os
import re


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

if __name__ == "__main__":
    config = load_config(path='config.json')
    main(config)