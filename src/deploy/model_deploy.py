import sys
import sklearn
import argparse
import os
from azureml.core import Workspace, Datastore, Environment
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import Model, InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.run import Run, _OfflineRun

model_name = "test_model"

print("Argument 1(model_name): %s" % model_name)

environmentName = "mlops-env"
serviceName = "test-model" + "-svc"
modelDirectory = model_name
modelName = model_name
entryScript = modelDirectory + "/raw/scoring_file.py"
modelFileName = modelDirectory + "/Temp/model.pkl"
modelPath = "files/" + modelDirectory + "/Temp"


def guardar_scoring_uri(run, endpoint_uri):
    """Upload to current Run the endpoint uri"""
    with open("scoring_uri.txt", "w") as f:
        f.write(f"{endpoint_uri}")
    run.upload_file(
        name="outputs/scoring_uri.txt",
        path_or_stream="scoring_uri.txt",
    )


run = Run.get_context()
ws = None
if type(run) == _OfflineRun:
    ws = Workspace.from_config()
else:
    ws = run.experiment.workspace

# dstor = Datastore.get(ws, datastore_name=datastoreName)
# dstor = ws.get_default_datastore()
# dstor.download('files', model_name, overwrite=True)

environment = Environment(environmentName)
with open("pipelines/req.txt", "r") as f:
    pip_packages = f.readlines()

conda_dependencies = CondaDependencies().create()
conda_dependencies.set_python_version("3.6.2")
conda_dependencies.add_conda_package("pip==20.2.4")
for pack in pip_packages:
    conda_dependencies.add_pip_package(pack)
environment.python.conda_dependencies = conda_dependencies

print("done")

inferenceConfig = InferenceConfig(
    entry_script="src/inference/scoring_file.py", environment=environment
)
# inferenceConfig = InferenceConfig(environment=environment)
aciConfig = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
model = Model(ws, model_name)
print("done")

service = Model.deploy(
    workspace=ws,
    name=serviceName.lower(),
    models=[model],
    inference_config=inferenceConfig,
    deployment_config=aciConfig,
    overwrite=True,
)
service.wait_for_deployment(show_output=True)
print(service.get_logs())

# Scoring uri
endpoint_uri = service.scoring_uri
guardar_scoring_uri(run, endpoint_uri)

print("done")
