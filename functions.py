import os
import math
import numpy as np

def delete_files(dir_path):
    """
    delete all files in the given folder
    :param dir_path: the path of the given folder
    :return:
    """
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))  # delete files
