import datetime
import os
import logging
from PIL import Image
import logging
from logging.handlers import TimedRotatingFileHandler
import  numpy as np

def filer(test):
    now = datetime.datetime.now()
    return '/data/log.txt'+now.strftime("%Y-%m-%d_%H:%M:%S")

def check_if_folder_exists(folder):
    if not os.path.exists(folder):
        logging.error(f"Folder {folder} does not exist")
        return False
    return True

def save_file(data):
    """Save file to /data/images
    Args:
        data (np.array): Image to save
    Returns:
        bool: True if file is saved, False otherwise
    """
    try:
        if not os.path.exists("/data/images"):
            os.makedirs("/data/images")
        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        data = data.astype(np.uint8)
        im = Image.fromarray(data)
        im.save(f"/data/images/{time}.png", "PNG")
        logging.info(f"File saved: {time}.png")
        return True
    except Exception as e:
        logging.error(f"Error saving file: {e}")
        logging.exception("Error saving file")
        return False
    
def setup_logging():
    """Setup logging
    Returns:
        logger: Logger object
    """
    # Create the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger = logging.getLogger("Rotating Log")
    logger.setLevel(logging.INFO)
    
    handler = TimedRotatingFileHandler("/data/app.log",
                                       when="d",
                                       interval=1)
    logger.addHandler(handler)

    return logger