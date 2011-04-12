"""
This is the top-level module of Luminoso version 2.
"""
from luminoso.model import LuminosoModel
import warnings
import logging, sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# suppress warnings internal to PyLint
warnings.simplefilter("ignore")
make_english = LuminosoModel.make_english
make_japanese = LuminosoModel.make_japanese
make_empty = LuminosoModel.make_empty

def load(model_dir):
    """
    Load a LuminosoModel.
    """
    return LuminosoModel(model_dir)
