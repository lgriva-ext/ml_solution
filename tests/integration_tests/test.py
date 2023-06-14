"""_summary_
"""
import pytest
from datetime import datetime
import numpy as np
import mlflow
from src.train_db import train


def test_integration_w_model_registry():
    # Check if registering a model, then will bring me access to it
    registry_uri = "test"
    mlflow.set_registry_uri(uri=registry_uri)

    mlflow.set_experiment(f"test_training")

    data_args = {"x_train": np.array(range(500)).reshape(500,1), "y_train": np.array(range(500,1000)), "x_val": np.array(range(100)).reshape(100,1), "y_val": np.array(range(500,600))}
    model_name = f"test_a_{datetime.now().strftime('%Y-%m-%d')}"
    train.model_name = model_name
    model_details = train.estimate_model(data_args=data_args)
    assert model_details.name == model_name
    assert model_details.current_stage == "None"


def test_xx():
    # Check if feature storing some features then will bring me access to they
    pass


def test_y():
    # Check all preprocess.py
    pass


def test_yy():
    # Check all train.py
    pass
