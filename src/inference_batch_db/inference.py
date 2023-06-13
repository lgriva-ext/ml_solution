"""_summary_
"""
import os
import json
import pandas as pd
import mlflow
from databricks import feature_store
from pyspark.sql.functions import monotonically_increasing_id


feature_store_uri = f"databricks://featurestore:featurestore"
fs = feature_store.FeatureStoreClient(feature_store_uri=feature_store_uri)
aux_path = "/".join(os.getcwd().split("/")[:-2])
config_path = f"{aux_path}/jobs/current_model.json"
model_name = json.load(open(config_path, "r"))["current_model_name"]
logged_model = f"models:/{model_name}/Production"


def addIdColumn(dataframe, id_column_name):
    """Add id column to dataframe"""
    columns = dataframe.columns
    new_df = dataframe.withColumn(id_column_name, monotonically_increasing_id())
    return new_df[[id_column_name] + columns]


if __name__ == "__main__":
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Load test data
    df_test = fs.read_table("test_data").toPandas()
    df_test = df_test[[c for c in df_test.columns if c != "item_id"]]

    # Predict on a Pandas DataFrame.
    preds = loaded_model.predict(df_test)
    preds_sdf = spark.createDataFrame(preds)
    preds_sdf_schema = preds_sdf.schema

    # Write predictions to feature store
    uid = "s"
    preds_sdf = addIdColumn(preds_sdf, "item_id")
    fs.create_table(
        name="predictions_" + uid,
        # df=preds_sdf,
        primary_keys=["item_id"],
        schema=preds_sdf_schema,
        description="raw test bigmart features",
    )
    fs.write_table(name="test_data_preprocessed_" + uid, df=preds_sdf, mode="overwrite")
