#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig

from azureml.core.runconfig import CondaDependencies, RunConfiguration

from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
import azureml.core
import pandas as pd
import numpy as np
import logging

from azureml.core.workspace import Workspace
from azureml.core.experiment import Experiment
from azureml.train.automl import AutoMLConfig
from azureml.automl.core.featurization import FeaturizationConfig

from azureml.pipeline.core import Pipeline, PipelineParameter, StepSequence
from azureml.pipeline.steps import PythonScriptStep

from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import Pipeline, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Schedule, ScheduleRecurrence

## Import and Workspace Creation

# subscription_id = '2fa38a83-8c49-4cae-90fc-346d90c21eb8'
# resource_group = 'airetail-latam'
# workspace_name = 'mlops'

# interactive_auth = InteractiveLoginAuthentication(tenant_id="30af61e6-f207-4ecc-97ac-2932bc0503dc", force=True)

# ws = Workspace(subscription_id=subscription_id,
#               resource_group=resource_group,
#               workspace_name=workspace_name,
#               auth=interactive_auth)

ws = Workspace.from_config()

## Experiment

experiment_name = "automl-orders-database"

experiment = Experiment(ws, experiment_name)

## Create Training Resource

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your CPU cluster
amlcompute_cluster_name = "latam-mlops-aml"

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=amlcompute_cluster_name)
    print("Found existing cluster, use it.")
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_D2_V2", max_nodes=6
    )
    compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True)

compute_target = ComputeTarget(workspace=ws, name="latam-mlops-aml")

# Environment

# create a new RunConfig object
conda_run_config = RunConfiguration(framework="python")

# Set compute target to AmlCompute
conda_run_config.target = compute_target

# use the azureml.core.runconfig.DockerConfiguration object with the 'use_docker' param instead
conda_run_config.environment.docker.enabled = True

cd = CondaDependencies.create(
    pip_packages=[
        "azureml-sdk[automl]",
        "applicationinsights",
        "azureml-opendatasets",
        "azureml-defaults",
    ],
    conda_packages=[
        "numpy==1.16.2",
        "pyodbc",
        "pandas==0.25.1",
        "scikit-learn==0.22.1",
        "holidays==0.9.11",
        "fbprophet==0.5",
        "psutil>=5.2.2",
        "py-xgboost<=0.90",
        "scipy",
    ],
    pin_sdk_version=False,
)
conda_run_config.environment.python.conda_dependencies = cd


dstor = ws.get_default_datastore()
# Choose a name for the run history container in the workspace.
experiment_name = "retrain-parkLotOcc-data"
experiment = Experiment(ws, experiment_name)

print("run config is ready")


# Construcción pipeline de entramiento y asignado de schedule (se entrena cada 7 días)

dataset = "BigMart"
ds_name = PipelineParameter(name="ds_name", default_value=dataset)
dstor = ws.get_default_datastore()
model_name = PipelineParameter("model_name", default_value="bigmart-reg")

preprocess_step = PythonScriptStep(
    script_name="preprocess/pre_process.py",
    allow_reuse=False,
    name="preprocess_step",
    arguments=["--ds_name", ds_name],
    compute_target=compute_target,
    runconfig=conda_run_config,
)

train_step = PythonScriptStep(
    script_name="train.py",
    name="train_step",
    allow_reuse=False,
    arguments=[
        "--model_name",
        model_name,
        "--ds_name",
        ds_name,
    ],  # "--model_path", model_data],
    # outputs=[model_data],
    compute_target=compute_target,
    runconfig=conda_run_config,
)

evaluate_model_step = PythonScriptStep(
    script_name="evaluate_model.py",
    name="evaluate_model",
    allow_reuse=False,
    arguments=[
        "--model_name",
        model_name,
        "--ds_name",
        ds_name,
    ],  # "--model_path", model_data],
    # inputs=[model_data],
    compute_target=compute_target,
    runconfig=conda_run_config,
)

register_model_step = PythonScriptStep(
    script_name="register_model.py",
    name="register_model",
    allow_reuse=False,
    arguments=["--model_name", model_name, "--ds_name", ds_name],
    compute_target=compute_target,
    runconfig=conda_run_config,
)

step_sequence = StepSequence(
    steps=[preprocess_step, train_step, evaluate_model_step, register_model_step]
)
## Training pipeline with all the steps
training_pipeline = Pipeline(
    description="training_pipeline", workspace=ws, steps=step_sequence
)

pipeline_name = "Retraining-Pipeline"

published_pipeline = training_pipeline.publish(
    name=pipeline_name, description="Pipeline that retrains AutoML model"
)

# published_pipeline

recurrence = ScheduleRecurrence(frequency="Day", interval=7)
schedule = Schedule.create(
    ws,
    name="TestSchedule",
    pipeline_id=published_pipeline.id,
    experiment_name=experiment_name,
    recurrence=recurrence,
    wait_for_provisioning=True,
)


# Construcción pipeline de deployado y asignado de schedule (se deploya cuando hay un movimiento en la carpeta model_name/Temp
# del default blob storage)

deploy_model_step = PythonScriptStep(
    script_name="deploy/model_deploy.py",
    allow_reuse=False,
    name="deploy_automl_parklot_database",
    arguments=["--model_name", model_name],
    compute_target=compute_target,
    runconfig=conda_run_config,
)

deploy_pipeline = Pipeline(
    description="pipeline_with_deploymodel", workspace=ws, steps=[deploy_model_step]
)

pipeline_name = "Deploying-Pipeline"

dpublished_pipeline = deploy_pipeline.publish(
    name=pipeline_name, description="Pipeline that deploys AutoML model"
)

schedule = Schedule.create(
    workspace=ws,
    name="DeployingSchedule",
    pipeline_parameters={"ds_name": dataset, "model_name": model_name},
    pipeline_id=dpublished_pipeline.id,
    experiment_name=experiment_name,
    datastore=dstor,
    path_on_datastore=model_name + "/Temp",
    wait_for_provisioning=True,
    polling_interval=5,
)
