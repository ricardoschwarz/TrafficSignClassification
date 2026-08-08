# Thin re-export shim. utils.py used to hold everything (labels, data loading, model
# factories, plotting); those responsibilities now live in labels.py, data.py, models.py
# and plotting.py respectively. This shim keeps `from utils import ...` working for any
# code (in this repo or elsewhere) still importing the old names from here.
from data import load_training_data
from labels import classes
from models import get_basic_model, get_complex_model
from plotting import (
    compute_confusion_matrix,
    plot_confusion_matrix,
    plot_history,
    show_images,
)

__all__ = [
    "classes",
    "load_training_data",
    "get_basic_model",
    "get_complex_model",
    "show_images",
    "plot_history",
    "plot_confusion_matrix",
    "compute_confusion_matrix",
]
