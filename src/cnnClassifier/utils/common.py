import os
from box.exceptions import BoxValueError
import yaml
from src.cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_ymal(path_to_ymal: Path) ->ConfigBox:
    """read ymal file and returns
    Args:
        path_to_ymal(str) :path like input

    Raises:
        ValueError: if ymal file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_ymal) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_ymal} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbos=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dir is to be created
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbos:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data
    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path,"w") as f:
        json.dump(data,f,indent=4)
    logger.info(f"json file saved at {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json file from the path
    Args:
        path (Path): path of the json file
    Returns:
        ConfigBox: ConfigBox type
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"json file loaded successfully from path: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file
    Args:
        data (Any): data to be saved as binary
        path: path of that binary file
        """
    joblib.dump(value=data,filename=path)
    logger.info(f"binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data
    Args:
        path (Path): path to binary file
    Retruns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded successfully from path: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB
    Args:
        path (Path): path of the file
    Returns:
        str: size of file in KB
    """
    size_of_file = round(os.path.getsize(path)/1024)
    return f"~ {size_of_file} KB"

def decodeImage(imgstring,fileName):
    """decode an image string
    Args:
        imgstring: image string
        fileName: name of the file
    """
    imgdata = base64.b64decode(imgstring)
    with open(fileName,'wb') as f:
        f.write(imgdata)
        f.close()

def encodeImageIntoBase64(croppedImagePath):
    """encode an image into base64
    Args:
        croppedImagePath: cropped image path
    Returns:
        Encoded binary Image
        """
    with open(croppedImagePath,"rb") as f:
        return base64.b16encode(f.read())

