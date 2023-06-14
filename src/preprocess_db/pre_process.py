import json
import os
from datetime import date
import pandas as pd
import logging
from databricks import feature_store
from pyspark.sql.functions import monotonically_increasing_id


ds_name = "ds_test"
logging.info("Pre-processing data.")


def get_feature_store():
    feature_store_uri = f"databricks://featurestore:featurestore"
    fs = feature_store.FeatureStoreClient(feature_store_uri=feature_store_uri)
    return fs


def addIdColumn(dataframe, id_column_name):
    """Add id column to dataframe"""
    columns = dataframe.columns
    new_df = dataframe.withColumn(id_column_name, monotonically_increasing_id())
    return new_df[[id_column_name] + columns]


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


def preprocess_feat_eng(df, config_path, train=True, dict_mode=None, cuts=None):
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


def write_preprocessed_data_to_fs(fs, name, data, uid=""):
    schema = data.schema
    fs.create_table(
        name=name + uid,
        # df=df_train,
        primary_keys=["item_id"],
        schema=schema,
        description="raw train bigmart features",
    )
    fs.write_table(name=name + uid, df=data, mode="overwrite")


def execute():
    aux_path = "/".join(os.getcwd().split("/")[:-1])
    config_path = f"{aux_path}/configs"
    fs = get_feature_store()

    df_train = fs.read_table("train_data_raw").toPandas()
    df_train = df_train[[c for c in df_train.columns if c != "item_id"]]
    df_test = fs.read_table("test_data_raw").toPandas()
    df_test = df_test[[c for c in df_test.columns if c != "item_id"]]

    df_train, dict_mode, l = preprocess_feat_eng(df_train, config_path)
    df_test, dict_mode, l = preprocess_feat_eng(
        df_test, config_path, False, dict_mode, l
    )

    df_train = spark.createDataFrame(df_train)
    df_test = spark.createDataFrame(df_test)

    df_train = addIdColumn(df_train, "item_id")
    df_test = addIdColumn(df_test, "item_id")

    uuid = json.loads(
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson()
    )["tags"]["jobId"]
    ddate = str(date.today()).replace("-", "_")
    uid = f"{uuid}_{ddate}"

    write_preprocessed_data_to_fs(fs, "train_data_preprocessed_", df_train, uid)
    write_preprocessed_data_to_fs(fs, "test_data_preprocessed_", df_test, uid)


if __name__ == "__main__":
    execute()

    logging.info("Finished Preprocessing")
