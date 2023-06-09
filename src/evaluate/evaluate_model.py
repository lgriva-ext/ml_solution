#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from azureml.core.model import Model, Dataset
from azureml.core.run import Run, _OfflineRun
from azureml.core import Workspace
import argparse
from azureml.core.resource_configuration import ResourceConfiguration
import os
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

from scipy import stats
from scipy.stats import wilcoxon

ds_name = "ds_test"
model_name = "test_model"

print("Argument 1(model_name): %s" % model_name)
# print("Argument 2(model_path): %s" % args.model_path)
print("Argument 2(ds_name): %s" % ds_name)

run = Run.get_context()
ws = None
if type(run) == _OfflineRun:
    ws = Workspace.from_config()
else:
    ws = run.experiment.workspace
    # print(dir(run))
# dstor = Datastore.get(ws, "testops_datastore")

# ds_name es igual a ""
# dstor.download('files', ds_name)
with open("datos/" + ds_name + "/conditions_retraining.txt", "r") as f:
    a = f.read()
f.close()

if a == "ft":  # if first_run pass too!!
    pass
else:
    umodel_previous_version = Model(ws, model_name)
    model_previous_version = pd.read_pickle(
        umodel_previous_version.download(exist_ok=True) + "/model.pkl"
    )
    # dstor.download('files', ds_name)
    d_test = pd.read_csv("datos/" + ds_name + "/val/data/Val_BigMart.csv")
    # model_new_version = pd.read_pickle(args.model_path)
    # dstor.download('files', 'Temp')

    folder_name = "Temp/{0}".format(model_name)
    # file_path = "{0}/model_datax".format(folder_name)
    file_path = "{0}/model.pkl".format(folder_name)
    model_new_version = pd.read_pickle(file_path)

    test_data = d_test.drop("Item_Outlet_Sales", axis=1)
    y_real = d_test["Item_Outlet_Sales"].values
    y_hat_previous = model_previous_version.predict(test_data)
    y_hat_new = model_new_version.predict(test_data)

    stat, pv = wilcoxon(y_hat_new[0] - y_real, y_hat_previous[0] - y_real)

    if not r2_score(y_hat_new[0], y_real) > r2_score(y_hat_previous[0], y_real):
        print("Cancelar run porque el nuevo modelo no es mejor que el anterior")
        run.parent.cancel()
    else:
        # New data is available since the model was last trained
        if pv < 0.05:
            print("El nuevo modelo será registrado")
        else:
            print("Cancelar run porque el nuevo modelo no es mejor que el anterior")
            run.parent.cancel()
