"""
This is the top-level module of Luminoso version 2.
"""
from luminoso.luminoso_space import LuminosoSpace
import warnings

# suppress warnings internal to PyLint
warnings.simplefilter("ignore")

def make_english(space_dir):
    """
    A shortcut to make a new study, trained on English-language common sense,
    in the given directory (which must not already exist).
    """
    return LuminosoSpace.make_english(space_dir)

def load(space_dir):
    """
    Load a LuminosoSpace.
    """
    return LuminosoSpace(space_dir)