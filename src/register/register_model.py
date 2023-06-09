import shutil
from azureml.core.model import Model, Dataset
from azureml.core.run import Run, _OfflineRun
from azureml.core import Workspace
import argparse
from azureml.core.resource_configuration import ResourceConfiguration
import os
import sklearn

ds_name = "ds_test"
model_name = "test_model"

print("Argument 1(model_name): %s" % model_name)
print("Argument 2(ds_name): %s" % ds_name)

run = Run.get_context()
ws = None
if type(run) == _OfflineRun:
    ws = Workspace.from_config()
else:
    ws = run.experiment.workspace
    print(dir(run))

# train_ds = Dataset.get_by_name(ws, ds_name + '_train')
# datasets = [(Dataset.Scenario.TRAINING, train_ds)]

# Register model with training dataset
# dstor = ws.get_default_datastore()
# dstor.download('files', ds_name + '/Temp')

dictt = dict()
dictt[f"datos/{ds_name}/cuts/cuts.txt"] = "/cuts/cuts.txt"
dictt[f"datos/{ds_name}/modas/modas.json"] = "/modas/modas.json"
dictt["data/prod_no_aplica.txt"] = "/prod_no_aplica.txt"
dictt["data/tiendas.txt"] = "/tiendas.txt"
for k in dictt:
    dest = "/".join(("Temp" + dictt[k]).split("/")[:-1])
    if not os.path.isdir(dest):
        os.makedirs(dest, exist_ok=True)
    shutil.copyfile(k, "Temp" + dictt[k])
model = Model.register(
    workspace=ws,
    model_path="Temp",  # + '/model.pkl',
    model_name=model_name,
    model_framework=Model.Framework.SCIKITLEARN,  # Framework used to create the model.
    model_framework_version=sklearn.__version__,
    resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=1),
    # datasets=datasets
)

# dstor.download('files', ds_name + '/raw')
#
# dstor.upload_files(files=['{0}/{1}/{2}/metrics_data.json'.format('files', ds_name, 'Temp'),
#                        '{0}/{1}/{2}/model.pkl'.format('files', ds_name, 'Temp'),
#                        '{0}/{1}/{2}/scoring_file.py'.format('files', ds_name, 'raw'),
#                        #'{0}/{1}/{2}/dependencies_file.yml'.format('files', ds_name, 'raw'),
#                        '{0}/{1}/{2}/modas/modas.json'.format('files', ds_name, 'Temp'),
#                        '{0}/{1}/{2}/cuts/cuts.txt'.format('files', ds_name, 'Temp')] ,  target_path=model_name, overwrite=True, show_progress=True)
# os.makedirs()
# model.download()

print("Registered version {0} of model {1}".format(model.version, model.name))
