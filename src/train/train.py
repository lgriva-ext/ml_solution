"""_summary_
"""
import argparse
import os
import azureml.core
from datetime import datetime
import pandas as pd
import pytz
from azureml.core import Dataset, Model
from azureml.core.run import Run, _OfflineRun
from azureml.core import Workspace
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import joblib
import pickle
import json

run = Run.get_context()
ws = None
if type(run) == _OfflineRun:
    ws = Workspace.from_config()
else:
    ws = run.experiment.workspace

print("Training")

ds_name = "ds_test"
model_name = "test_model"

# dstor = Datastore.get(ws, "testops_datastore")
# dstor.download('datos', ds_name + '/preprocessed')

seed = 28
model = LinearRegression()

df_train = pd.read_csv("datos/" + ds_name + "/preprocessed/data/Train_BigMart.csv")
df_train.dropna(inplace=True)
# dataset = Dataset.get_by_name(ws, name=ds_name + '_train')
# df_train = dataset.to_pandas_dataframe()

# División de dataset de entrenaimento y validación
X = df_train.drop(
    columns="Item_Outlet_Sales"
)  # [['Item_Weight', 'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type']] # .drop(columns='Item_Outlet_Sales')
x_train, x_val, y_train, y_val = train_test_split(
    X, df_train["Item_Outlet_Sales"], test_size=0.3, random_state=seed
)
df_val = pd.DataFrame(pd.concat([x_val, y_val], axis=1))

# Entrenamiento del modelo
model.fit(x_train, y_train)

# Predicción del modelo ajustado para el conjunto de validación
pred = model.predict(x_val)

# Cálculo de los errores cuadráticos medios y Coeficiente de Determinación (R^2)
mse_train = metrics.mean_squared_error(y_train, model.predict(x_train))
R2_train = model.score(x_train, y_train)
print("Métricas del Modelo:")
print("ENTRENAMIENTO: RMSE: {:.2f} - R2: {:.4f}".format(mse_train**0.5, R2_train))

mse_val = metrics.mean_squared_error(y_val, pred)
R2_val = model.score(x_val, y_val)
val_metrics = dict()
val_metrics["mse_val"] = mse_val
val_metrics["R2_val"] = R2_val
print("VALIDACIÓN: RMSE: {:.2f} - R2: {:.4f}".format(mse_val**0.5, R2_val))

folder_name = "Temp/{0}".format(model_name)
file_path = "{0}/model.pkl".format(folder_name)
file_path1 = "{0}/metrics_data.json".format(folder_name)
os.makedirs(folder_name, exist_ok=True)

with open(file_path1, "w") as json_file:
    json.dump(val_metrics, json_file)
joblib.dump(model, open(file_path, "wb"))
# pickle.dump(fitted_model, open(file_path, 'wb'))

datos_val_folder = "datos/" + ds_name + "/val/data/"
os.makedirs(datos_val_folder, exist_ok=True)
df_val.to_csv(datos_val_folder + "/Val_BigMart.csv", index=None, header=True)

# dstor.upload_files(files=[file_path, file_path1], target_path=ds_name + '/Temp', overwrite=True, show_progress=True)
# dstor.upload_files(files=[datos_val_folder + '/Val_BigMart.csv'], target_path=ds_name + '/val/data', overwrite=True, show_progress=True)

print("Finished Training")
