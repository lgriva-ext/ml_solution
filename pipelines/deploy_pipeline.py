"""
Code to execute the azure ml pipeline to serve registered model.
"""
# Importing standard libraries.
import os

# Importing azure ml libraries.
# from azureml.pipeline.core._restclients.aeva.models.error_response import (
#    ErrorResponseException,
# )
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.core import Workspace, Environment, Experiment
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.authentication import ServicePrincipalAuthentication


# Environmental variables needed to have the authorization to run on azure ml Workspace
tenant_id = os.getenv("smdc_tenant_id")
client_id = os.getenv("smdc_client_id")
client_secret = os.getenv("smdc_client_secret")
compute_cluster_name = os.getenv("compute_cluster_name")
auth = ServicePrincipalAuthentication(
    tenant_id=tenant_id,
    service_principal_id=client_id,
    service_principal_password=client_secret,
)

# Azure ml Workspace obtention
ws = Workspace.get(
    subscription_id="xxxx",
    resource_group="xxxx",
    name="xxxx",
    auth=auth,
)

# Compute target where pipeline will run
aml_compute_target = ComputeTarget(workspace=ws, name=compute_cluster_name)

# Environment
environment_variables = {
    "smdc_tenant_id": tenant_id,
    "smdc_client_id": client_id,
    "smdc_client_secret": client_secret,
}

# Libraries needed
with open("pipelines/req.txt", "r") as f:
    pip_packages = f.readlines()

env = Environment(name="myenv")
conda_dependencies = CondaDependencies().create()
conda_dependencies.set_python_version("3.6.2")
conda_dependencies.add_conda_package("pip==20.2.4")
for pack in pip_packages:
    conda_dependencies.add_pip_package(pack)
env.python.conda_dependencies = conda_dependencies

# Run configuration with defined compute target and environment
aml_run_config = RunConfiguration()
aml_run_config.target = aml_compute_target
aml_run_config.node_count = 1
aml_run_config.environment = env
aml_run_config.environment_variables = environment_variables

# Arguments for pipeline step creation
kwargs_modelling = {
    "name": "run_deploying_pipeline",
    "script_name": "./src/deploy/model_deploy.py",
    "source_directory": f"{os.getcwd()}",
    "compute_target": aml_compute_target,
    "runconfig": aml_run_config,
    "allow_reuse": False,
}
# Step object
run_deploying_pipeline = PythonScriptStep(**kwargs_modelling)

# Pipeline object
pipeline = Pipeline(
    workspace=ws,
    steps=[run_deploying_pipeline],
    default_source_directory=f"{os.getcwd()}",
)

# Run pipeline and wait for completion
experiment = Experiment(ws, "test")
pipeline_run_exp = experiment.submit(pipeline)
pipeline_run_exp.wait_for_completion()

# Downloading file with scoring uri
for pipeline_run in pipeline_run_exp.get_steps():
    pipeline_run.download_file("outputs/scoring_uri.txt")
    pipeline_run.download_file("user_logs/std_log.txt")

# Set scoring uri as environmental variable to show it on github action workflow
env_file = os.getenv("GITHUB_ENV")
if env_file:
    with open("scoring_uri.txt", "r") as f:
        endpoint_uri = f.read()
    with open(env_file, "a") as f:
        f.write(f"SCORING_URI={endpoint_uri}")
