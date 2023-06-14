"""_summary_
"""
import pytest
from datetime import datetime
import pandas as pd
import numpy as np
import mlflow
from src.train_db import train
from src.preprocess_db import pre_process
from databricks import feature_store


def test_integration_w_model_registry():
    # Check if registering a model, then will bring me access to it
    registry_uri = "test"
    mlflow.set_registry_uri(uri=registry_uri)

    mlflow.set_experiment(f"test_training")

    data_args = {
        "x_train": np.array(range(500)).reshape(500, 1),
        "y_train": np.array(range(500, 1000)),
        "x_val": np.array(range(100)).reshape(100, 1),
        "y_val": np.array(range(500, 600)),
    }
    model_name = f"test_a_{datetime.now().strftime('%Y-%m-%d')}"
    train.model_name = model_name
    model_details = train.estimate_model(data_args=data_args)
    assert model_details.name == model_name
    assert model_details.current_stage == "None"


def test_integration_w_feature_store():
    # Check if feature storing some features then will bring me access to they
    fs = feature_store.FeatureStoreClient()
    df = pd.DataFrame(data=[np.array(range(500))], columns=["x"])
    df.loc[:, "y"] = np.array(range(500, 1000)).tolist()
    df.reset_index(inplace=True)
    df.rename(columns={"index": "item_id"}, inplace=True)
    df = spark.createDataFrame(df)
    pre_process.write_preprocessed_data_to_fs(fs, "test", df)
    dff = fs.read_table("test").toPandas()
    assert dff.shape == df.shape
    for c in df.columns:
        assert c in dff.columns


def test_y():
    # Check all preprocess.py
    pass


def test_yy():
    # Check all train.py
    pass


if __name__ == "__main__":
    test_y()
    test_yy()
    test_integration_w_model_registry()
    test_integration_w_feature_store()
