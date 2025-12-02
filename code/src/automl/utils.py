# # Copyright (C) KonaAI - All Rights Reserved
"""This module loads all underlying constants"""
import configparser
import os
from typing import Any

config_file_path = os.path.join(os.path.dirname(__file__), "config.ini")
config = configparser.ConfigParser()
config.optionxform = str
config.read(config_file_path, encoding="utf-8")


def config_save(section: str, option: Any, value: Any):
    """This function is used to save the configuration"""
    config.set(section, str(option), str(value))
    with open(config_file_path, "w", encoding="utf-8") as configfile:
        config.write(configfile)
