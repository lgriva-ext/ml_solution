"""_summary_
"""
import requests
from datetime import date
import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import json
import logging
import mlflow
from databricks import feature_store
from pyspark.sql.functions import monotonically_increasing_id
import argparse


### POSSIBLY SET THIS EXPERIMENT URI AS A SECRET
### THEN DOWNLOAD FROM THERE THE ARTIFACTS TO CHECK COMPLIANCE!!
ds_name = "ds_test"
model_name = "test_model"
seed = 28

logging.info("Training")


def get_arguments():
    parser = argparse.ArgumentParser(description="reads default arguments")
    parser.add_argument(
        "--run_id", metavar="run_id", type=str, help="Databricks run id"
    )
    parser.add_argument(
        "--job_id", metavar="job_id", type=str, help="Databricks job id"
    )
    args = parser.parse_args()

    return args.run_id, args.job_id


def set_model_registry():
    registry_uri = f"databricks://modelregistery:modelregistery"
    mlflow.set_registry_uri(registry_uri)

    try:
        uuid = json.loads(
            dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson()
        )["tags"]["jobId"]
        ddate = str(date.today()).replace("-", "_")
        uid = f"{uuid}_{ddate}"
    except:
        uid = "a"

    mlflow.set_experiment(f"/Shared//test_training_{uid}")

    return uid


def addIdColumn(dataframe, id_column_name):
    """Add id column to dataframe"""
    columns = dataframe.columns
    new_df = dataframe.withColumn(id_column_name, monotonically_increasing_id())
    return new_df[[id_column_name] + columns]


def transition_to_staging(name, version):
    job_payload = {
        "name": name,
        "version": version,
        "stage": "Staging",
        "archive_existing_versions": False,
        "comment": "Staging version of this model",
    }

    resp = requests.post(
        f'{dbutils.secrets.get(scope="modelregistery", key="modelregistery-host")}api/2.0/mlflow/databricks/model-versions/transition-stage',
        json=job_payload,
        headers={
            "Authorization": f'Bearer {dbutils.secrets.get(scope="modelregistery", key="modelregistery-token")}'
        },
    )

    print(resp.status_code)


def request_transition_to_staging(name, version):
    job_payload = {
        "name": name,
        "version": version,
        "stage": "Staging",
        "comment": "Staging version of this model",
    }

    resp = requests.post(
        f'{dbutils.secrets.get(scope="modelregistery", key="modelregistery-host")}api/2.0/mlflow/transition-requests/create',
        json=job_payload,
        headers={
            "Authorization": f'Bearer {dbutils.secrets.get(scope="modelregistery", key="modelregistery-token")}'
        },
    )

    print(resp.status_code)


def estimate_model(data_args):
    """_summary_

    Args:
        data_args (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_train = data_args["x_train"]
    y_train = data_args["y_train"]
    x_val = data_args["x_val"]
    y_val = data_args["y_val"]

    with mlflow.start_run():
        # Automatically capture the model's parameters, metrics, artifacts,
        # and source code with the `autolog()` function
        mlflow.sklearn.autolog()

        model = LinearRegression()
        # Entrenamiento del modelo
        model.fit(x_train, y_train)

        run_id = mlflow.active_run().info.run_id

        tags = {
            "GIT_SHA": git_sha,
            "MLFLOW_RUN_ID": run_id,
            "DBR_JOB_ID": job_id,
            "DBR_RUN_ID": run_id,
        }
        mlflow.set_tags(tags)

    # Predicción del modelo ajustado para el conjunto de validación
    y_pred_val = model.predict(x_val)
    y_pred_train = model.predict(x_train)

    # Cálculo de los errores cuadráticos medios y Coeficiente de Determinación (R^2)
    mse_train = metrics.mean_squared_error(y_train, y_pred_train)
    R2_train = model.score(x_train, y_train)
    print("Métricas del Modelo:")
    print("ENTRENAMIENTO: RMSE: {:.2f} - R2: {:.4f}".format(mse_train**0.5, R2_train))

    mse_val = metrics.mean_squared_error(y_val, y_pred_val)
    R2_val = model.score(x_val, y_val)
    val_metrics = dict()
    val_metrics["mse_val"] = mse_val
    val_metrics["R2_val"] = R2_val
    print("VALIDACIÓN: RMSE: {:.2f} - R2: {:.4f}".format(mse_val**0.5, R2_val))

    artifact_path = "model"
    model_uri = "runs:/{run_id}/{artifact_path}".format(
        run_id=run_id, artifact_path=artifact_path
    )

    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

    return model_details


def get_feature_store():
    feature_store_uri = f"databricks://featurestore:featurestore"
    fs = feature_store.FeatureStoreClient(feature_store_uri=feature_store_uri)
    return fs


def execute():
    # model registry
    uid = set_model_registry()
    # fs
    fs = get_feature_store()
    # Reading
    df_train = fs.read_table(f"train_data_preprocessed_{uid}").toPandas()
    df_train = df_train[[c for c in df_train.columns if c != "item_id"]]

    df_train.dropna(inplace=True)

    # División de dataset de entrenaimento y validación
    X = df_train.drop(
        columns="Item_Outlet_Sales"
    )  # [['Item_Weight', 'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type']] # .drop(columns='Item_Outlet_Sales')
    X_cols = [c for c in df_train.columns if c != "Item_Outlet_Sales"]
    x_train, x_val, y_train, y_val = train_test_split(
        X, df_train["Item_Outlet_Sales"], test_size=0.3, random_state=seed
    )
    df_val = pd.DataFrame(
        data=pd.concat([x_val, y_val], axis=1).values, columns=df_train.columns
    )
    df_trained = pd.DataFrame(
        data=pd.concat([x_train, y_train], axis=1).values, columns=df_train.columns
    )

    data_args = {}
    data_args["x_train"] = x_train
    data_args["y_train"] = y_train
    data_args["x_val"] = x_val
    data_args["y_val"] = y_val
    model_details = estimate_model(data_args=data_args)

    df_train = addIdColumn(spark.createDataFrame(df_trained), "item_id")
    df_train_schema = df_train.schema
    df_val = addIdColumn(spark.createDataFrame(df_val), "item_id")
    df_val_schema = df_val.schema

    fs.create_table(
        name="train_data_" + uid,
        df=df_train,
        primary_keys=["item_id"],
        schema=df_train_schema,
        description="raw train bigmart features",
    )
    fs.create_table(
        name="val_data_" + uid,
        df=df_val,
        primary_keys=["item_id"],
        schema=df_val_schema,
        description="raw test bigmart features",
    )

    # transition_to_staging(model_details.name, model_details.version)
    request_transition_to_staging(model_details.name, model_details.version)


if __name__ == "__main__":
    #
    execute()
    run_id, job_id = get_arguments()
    git_sha = os.environ["GIT_SHA"]

    print("Finished Training", f"{run_id}", f"{job_id}", f"{git_sha}")
