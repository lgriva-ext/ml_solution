import json
import argparse
import os
import azureml.core
from datetime import datetime
import pandas as pd
import pytz
from azureml.core import Dataset, Model, Datastore
from azureml.data.datapath import DataPath
from azureml.core.run import Run, _OfflineRun
from azureml.core import Workspace

run = Run.get_context()
ws = None
if type(run) == _OfflineRun:
    ws = Workspace.from_config()
else:
    ws = run.experiment.workspace

print("Pre-processing data.")


def switcher(x, cuts):
    if x <= cuts[1]:
        return 1
    elif x > cuts[1] and x <= cuts[2]:
        return 2
    elif x > cuts[2] and x <= cuts[3]:
        return 3
    else:
        return 4


def item_mrp(df, train=True):
    print(df)
    if train == True:
        a = pd.qcut(df["Item_MRP"], 4).unique()
        df["Item_MRP"] = pd.qcut(df["Item_MRP"], 4, labels=[1, 2, 3, 4])
        os.makedirs("datos/" + ds_name + "/cuts", exist_ok=True)
        with open("datos/" + ds_name + "/cuts/cuts.txt", "w") as f:
            for aa in range(len(a)):
                f.write(str(a[aa].left) + "\n")
                if aa == len(a) - 1:
                    f.write(str(a[aa].right))
                else:
                    f.write(str(a[aa].right) + "\n")
        f.close()
    else:
        with open("datos/" + ds_name + "/cuts/cuts.txt", "r") as file2:
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


def mode_dict(df, train=True):
    dict_mode = dict()
    if train == True:
        productos = list(df[df["Item_Weight"].isnull()]["Item_Identifier"].unique())
        dfx = df[["Item_Identifier", "Item_Weight"]]
        # import pdb; pdb.set_trace()
        dfx = dfx.fillna(dfx.groupby("Item_Identifier")["Item_Weight"].transform(f))
        df["Item_Weight"] = dfx["Item_Weight"]
        for producto in productos:
            try:
                dict_mode[producto] = (
                    (df[df["Item_Identifier"] == producto][["Item_Weight"]])
                    .mode()
                    .iloc[0, 0]
                )
            except:
                dict_mode[producto] = (df[["Item_Weight"]]).mode().iloc[0, 0]
        os.makedirs("datos/" + ds_name + "/modas", exist_ok=True)
        with open("datos/" + ds_name + "/modas/modas.json", "w") as json_file1:
            json.dump(dict_mode, json_file1)
    else:
        dict_mode = json.load(open("datos/" + ds_name + "/modas/modas.json", "r"))
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


def preprocess_feat_eng(df, train=True):
    df["Outlet_Establishment_Year"] = 2020 - df["Outlet_Establishment_Year"]

    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(
        {"low fat": "Low Fat", "LF": "Low Fat", "reg": "Regular"}
    )

    with open("data/prod_no_aplica.txt", "r") as file1:
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

    print(df)
    if train == True:
        df = mode_dict(df)
        print("done train mode")
    else:
        df = mode_dict(df, False)
        print("done test mode")

    print(df)
    with open("data/tiendas.txt", "r") as ff:
        lines = ff.readlines()
        lista_tiendas = [line.rstrip() for line in lines]
    print(lista_tiendas)
    # for outlet in lista_tiendas:
    df.loc[df["Outlet_Identifier"].isin(lista_tiendas), "Outlet_Size"] = "Small"

    if train == True:
        df = item_mrp(df)
        print("done train item_mrp")
    else:
        df = item_mrp(df, False)
        print("done test item_mrp")

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


ds_name = "ds_test"

# La versión puede ser un parámetro también... entonces va a descargar la carpeta con esa versión de datos
dstor = Datastore.get(ws, "testops_datastore")
# dstor.download('datos', ds_name)

# Supongo que los datasets de train y test van a guardarse por un proceso de ETL por separado en una misma carpeta en el blob storage default.
# A su vez hay allí guardado un par de archivos que contienen los productos a los cuales no aplica el Fat_Content, las tiendas para las cuales
# se usará por defecto el tamaño 'Small'.
df_train = pd.read_csv("data/Train_BigMart.csv")
df_test = pd.read_csv("data/Test_BigMart.csv")

df_train = preprocess_feat_eng(df_train)
df_test = preprocess_feat_eng(df_test, False)

if not os.path.isdir("datos/" + ds_name + "/preprocessed/data"):
    os.makedirs("datos/" + ds_name + "/preprocessed/data", exist_ok=True)
df_train.to_csv(
    "datos/" + ds_name + "/preprocessed/data/Train_BigMart.csv", index=None, header=True
)
df_test.to_csv(
    "datos/" + ds_name + "/preprocessed/data/Test_BigMart.csv", index=None, header=True
)
alpha = False
try:
    ds = Dataset.get_by_name(ws, ds_name + "_train")
    # alpha = True
    with open("datos/" + ds_name + "/conditions_retraining.txt", "w") as f:
        f.write("la")
except:
    with open("datos/" + ds_name + "/conditions_retraining.txt", "w") as f:
        f.write("ft")
f.close()

folder_name = ds_name + "/preprocessed/data"
# dstor.upload_files(files=['datos/' + ds_name + '/preprocessed/data/Train_BigMart.csv'], target_path=folder_name, overwrite=True, show_progress=True)
# dstor.upload_files(files=['datos/' + ds_name + '/preprocessed/data/Test_BigMart.csv'], target_path=folder_name, overwrite=True, show_progress=True)

# ds = Dataset.Tabular.from_delimited_files(dstor.path("{}/Train_BigMart.csv".format(folder_name)))
# ds.register(ws, name=ds_name + '_train', create_new_version=True)
# ds = Dataset.Tabular.from_delimited_files(dstor.path("{}/Test_BigMart.csv".format(folder_name)))
# ds.register(ws, name=ds_name + '_test', create_new_version=True)

# dstor.upload_files(files=['datos/' + ds_name + '/conditions_retraining.txt'], target_path=folder_name + '/conditions_retraining', overwrite=True, show_progress=True)
# dstor.upload_files(files=['datos/' + ds_name + '/cuts/cuts.txt', 'datos/' + ds_name + '/modas/modas.json'], target_path=ds_name + '/Temp', overwrite=True, show_progress=True)

print("Finished Preprocessing")
