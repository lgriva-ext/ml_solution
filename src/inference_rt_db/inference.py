import os
import requests
import numpy as np
import pandas as pd
import json


def create_tf_serving_json(data):
    return {
        "inputs": {name: data[name].tolist() for name in data.keys()}
        if isinstance(data, dict)
        else data.tolist()
    }


def score_model(dataset):
    url = f'{os.getenv("DATABRICKS_HOST")}/serving-endpoints/test_model_ep/invocations'
    headers = {
        "Authorization": f'Bearer {os.environ.get("DATABRICKS_TOKEN")}',
        "Content-Type": "application/json",
    }
    ds_dict = (
        {"dataframe_split": dataset.to_dict(orient="split")}
        if isinstance(dataset, pd.DataFrame)
        else create_tf_serving_json(dataset)
    )
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method="POST", headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )
    return response.json()


def switcher(x, cuts):
    if x <= cuts[1]:
        return 1
    elif x > cuts[1] and x <= cuts[2]:
        return 2
    elif x > cuts[2] and x <= cuts[3]:
        return 3
    else:
        return 4


def item_mrp(df, train=True, cuts=None):
    print(df)
    if train == True:
        # TODO: Los archivos modas y cuts los genero de nuevo en el momento de registrar el modelo!
        a = pd.qcut(df["Item_MRP"], 4).unique()
        df["Item_MRP"] = pd.qcut(df["Item_MRP"], 4, labels=[1, 2, 3, 4])

        l = []
        for aa in range(len(a)):
            l.append(str(a[aa].left))
            if aa == len(a) - 1:
                l.append(str(a[aa].right))
            else:
                l.append(str(a[aa].right))
        return df, l
    else:
        cuts = list(dict.fromkeys(cuts))
        cuts = [float(aa) for aa in cuts]
        cuts.sort()
        df["Item_MRP"] = df["Item_MRP"].apply(switcher, args=(cuts,))
        return df, cuts


def f(x):
    try:
        return x.mode().iloc[0]
    except:
        return 0


def mode_dict(df, train=True, dict_mode=None):
    dict_mode = dict()
    if train == True:
        productos = list(df[df["Item_Weight"].isnull()]["Item_Identifier"].unique())
        dfx = df[["Item_Identifier", "Item_Weight"]]
        # TODO: check this
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

        return df, dict_mode
    else:
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
        return df, dict_mode


def preprocess_feat_eng(df, train=True, dict_mode=None, cuts=None):
    df["Outlet_Establishment_Year"] = 2020 - df["Outlet_Establishment_Year"]

    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(
        {"low fat": "Low Fat", "LF": "Low Fat", "reg": "Regular"}
    )

    lista_prod_no_aplica = json.load(
        open(f"{config_path}/preprocess_config.json", "r")
    )["prod_no_aplica"]

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

    if train == True:
        df, dict_mode = mode_dict(df)
        print("done train mode")
    else:
        df, dict_mode = mode_dict(df, False, dict_mode)
        print("done test mode")

    lista_tiendas = json.load(open(f"{config_path}/preprocess_config.json", "r"))[
        "tiendas"
    ]

    print(lista_tiendas)
    # for outlet in lista_tiendas:
    df.loc[df["Outlet_Identifier"].isin(lista_tiendas), "Outlet_Size"] = "Small"

    if train == True:
        df, l = item_mrp(df)
        print("done train item_mrp")
    else:
        df, l = item_mrp(df, False, cuts)
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
        df["Outlet_Type" + "_" + outlet_type.replace(" ", "_")] = df[
            "Outlet_Type"
        ].apply(lambda x: 1 if x == outlet_type else 0)

    df.drop(columns=["Outlet_Type"], inplace=True)
    df = df.drop(
        columns=[
            "Item_Identifier",
            "Outlet_Identifier",
            "Item_Type",
            "Item_Fat_Content",
        ]
    ).copy()

    return df, dict_mode, l


if __name__ == "__main__":
    print(score_model(dataset))
