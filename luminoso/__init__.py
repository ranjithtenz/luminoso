"""
This is the top-level module of Luminoso version 2.
"""
from luminoso.model import LuminosoModel
import warnings
import logging, sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# suppress warnings internal to PyLint
warnings.simplefilter("ignore")

def make_english(model_dir):
    """
    A shortcut to make a new study, trained on English-language common sense,
    in the given directory (which must not already exist).
    """
    return LuminosoModel.make_english(model_dir)

def load(model_dir):
    """
    Load a LuminosoModel.
    """
    return LuminosoModel(model_dir)
