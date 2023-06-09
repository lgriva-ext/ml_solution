# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


# input_sample = pd.DataFrame({"Item_Weight": pd.Series(19.10, dtype="float"), "Item_Visibility": pd.Series(19.10, dtype="float"), "Item_MRP": pd.Series(2, dtype="int"), "Outlet_Establishment_Year": pd.Series(2, dtype="int"), "Outlet_Size": pd.Series(2, dtype="int"), "Outlet_Location_Type": pd.Series(2, dtype="int"), "Outlet_Type_Grocery Store": pd.Series(2, dtype="int"), "Outlet_Type_Supermarket Type1": pd.Series(2, dtype="int"), "Outlet_Type_Supermarket Type2": pd.Series(2, dtype="int"), "Outlet_Type_Supermarket Type3": pd.Series(2, dtype="int")})
input_sample = pd.DataFrame(
    {
        "Item_Identifier": pd.Series("FDW58", dtype="str"),
        "Item_Weight": pd.Series(19.10, dtype="float"),
        "Item_Fat_Content": pd.Series("Low", dtype="str"),
        "Item_Visibility": pd.Series(19.10, dtype="float"),
        "Item_Type": pd.Series("Snacks", dtype="str"),
        "Item_MRP": pd.Series(2, dtype="int"),
        "Outlet_Identifier": pd.Series("OUT049", dtype="str"),
        "Outlet_Establishment_Year": pd.Series(2, dtype="int"),
        "Outlet_Size": pd.Series("Small", dtype="str"),
        "Outlet_Location_Type": pd.Series("Tier 1", dtype="str"),
        "Outlet_Type": pd.Series("Grocery", dtype="str"),
    }
)
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity("INFO")
    logger = logging.getLogger("azureml.automl.core.scoring_script_predicting")
except Exception:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "Temp/test_model/model.pkl"
    )
    print(os.getenv("AZUREML_MODEL_DIR"))
    # for root, dirs, files in os.walk(os.getenv('AZUREML_MODEL_DIR')):
    # for filename in files:
    # print(filename)
    print("model path esssss: {}".format(model_path))
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions(
        {"model_name": path_split[-3], "model_version": path_split[-2]}
    )
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


# primero pruebo haciendo el preprocess en el archivo que llama a la api
def switcher(x, cuts):
    if x <= cuts[1]:
        return 1
    elif x > cuts[1] and x <= cuts[2]:
        return 2
    elif x > cuts[2] and x <= cuts[3]:
        return 3
    else:
        return 4


def item_mrp(df):
    with open(
        os.path.join(os.getenv("AZUREML_MODEL_DIR"), "Temp/cuts/cuts.txt"), "r"
    ) as file2:
        cuts = file2.readlines()
        cuts = [line.rstrip() for line in cuts]
    cuts = list(dict.fromkeys(cuts))
    cuts = [float(aa) for aa in cuts]
    cuts.sort()
    df["Item_MRP"] = df["Item_MRP"].apply(switcher, args=(cuts,))
    return df


def f(x):
    try:
        return x.mode().iloc[0]
    except:
        return 0


def mode_dict(df):
    dict_mode = dict()
    dict_mode = json.load(
        open(os.path.join(os.getenv("AZUREML_MODEL_DIR"), "Temp/modas/modas.json"), "r")
    )
    productos = list(df[df["Item_Weight"].isnull()]["Item_Identifier"].unique())
    for producto in productos:
        try:
            df.loc[
                (df["Item_Identifier"] == producto) & (df["Item_Weight"].isnull()),
                "Item_Weight",
            ] = float(dict_mode[producto])
        except:
            moda = (df[["Item_Weight"]]).mode().iloc[0, 0]
            df.loc[
                (df["Item_Identifier"] == producto) & (df["Item_Weight"].isnull()),
                "Item_Weight",
            ] = moda
    return df


def preprocess_feat_eng(df):
    df["Outlet_Establishment_Year"] = 2020 - df["Outlet_Establishment_Year"]

    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(
        {"low fat": "Low Fat", "LF": "Low Fat", "reg": "Regular"}
    )

    with open(
        os.path.join(os.getenv("AZUREML_MODEL_DIR"), "Temp/prod_no_aplica.txt"), "r"
    ) as file1:
        lista_prod_no_aplica = file1.readlines()
        lista_prod_no_aplica = [line.rstrip() for line in lista_prod_no_aplica]
    print(lista_prod_no_aplica)
    for prod in lista_prod_no_aplica:
        df.loc[df["Item_Type"] == prod, "Item_Fat_Content"] = "NA"

    df["Item_Type"] = df["Item_Type"].replace(
        {
            "Others": "Non perishable",
            "Health and Hygiene": "Non perishable",
            "Household": "Non perishable",
            "Seafood": "Meats",
            "Meat": "Meats",
            "Baking Goods": "Processed Foods",
            "Frozen Foods": "Processed Foods",
            "Canned": "Processed Foods",
            "Snack Foods": "Processed Foods",
            "Breads": "Starchy Foods",
            "Breakfast": "Starchy Foods",
            "Soft Drinks": "Drinks",
            "Hard Drinks": "Drinks",
            "Dairy": "Drinks",
        }
    )

    # FEATURES ENGINEERING: asignación de nueva categorías para 'Item_Fat_Content'
    df.loc[df["Item_Type"] == "Non perishable", "Item_Fat_Content"] = "NA"

    df = mode_dict(df)

    print(df)
    with open(
        os.path.join(os.getenv("AZUREML_MODEL_DIR"), "Temp/tiendas.txt"), "r"
    ) as ff:
        lines = ff.readlines()
        lista_tiendas = [line.rstrip() for line in lines]
    print(lista_tiendas)
    # for outlet in lista_tiendas:
    df.loc[df["Outlet_Identifier"].isin(lista_tiendas), "Outlet_Size"] = "Small"

    df = item_mrp(df)

    df["Outlet_Size"] = df["Outlet_Size"].replace({"High": 2, "Medium": 1, "Small": 0})
    df["Outlet_Location_Type"] = df["Outlet_Location_Type"].replace(
        {"Tier 1": 2, "Tier 2": 1, "Tier 3": 0}
    )

    for outlet_type in [
        "Grocery Store",
        "Supermarket Type1",
        "Supermarket Type2",
        "Supermarket Type3",
    ]:
        df["Outlet_Type" + "_" + outlet_type] = df["Outlet_Type"].apply(
            lambda x: 1 if x == outlet_type else 0
        )

    df.drop(columns=["Outlet_Type"], inplace=True)
    df = df.drop(
        columns=[
            "Item_Identifier",
            "Outlet_Identifier",
            "Item_Type",
            "Item_Fat_Content",
        ]
    ).copy()

    return df


@input_schema("data", PandasParameterType(input_sample, enforce_shape=False))
def run(data):
    # data = json.loads(data)
    # if len(data) == 1:
    print(data)
    print(type(data))
    data = preprocess_feat_eng(data)
    data = data.values.reshape(1, -1)
    try:
        result = model.predict(data)
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})

    predict_as_list = result[0].tolist()
    # index_as_df = result[1].index.to_frame().reset_index(drop=True)

    return json.dumps(
        {
            "predict": predict_as_list  # ,   # return the minimum over the wire:
            # "index": json.loads(index_as_df.to_json(orient='records'))  # no predict and its featurized values
        }
    )
