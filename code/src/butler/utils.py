# # Copyright (C) KonaAI - All Rights Reserved
"""This module loads all underlying constants"""
import configparser
import os

config_file_path = os.path.join(os.path.dirname(__file__), "config.ini")
config = configparser.ConfigParser()
config.optionxform = str
config.read(config_file_path, encoding="utf-8")
