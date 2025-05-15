import sys
import os


def get_last_file_name(path: str):
    return os.path.split(path)[-1]