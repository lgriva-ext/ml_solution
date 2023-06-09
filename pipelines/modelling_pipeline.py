"""
Code to execute the azure ml modelling pipeline (feat eng, modelling, evaluation and registering).
"""
# Importing standard libraries.
import os

# Importing azure ml libraries.
from azureml.pipeline.core import Pipeline, PublishedPipeline, PipelineEndpoint
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
subs_id = os.getenv("azure_subs_id")
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
aml_compute_target = ComputeTarget(workspace=ws, name="xxxx")

# Environment
environment_variables = {
    "smdc_tenant_id": tenant_id,
    "smdc_client_id": client_id,
    "smdc_client_secret": client_secret,
}

# Libraries needed
with open("pipelines/req.txt", "r") as f:
    pip_packages = f.readlines()

env = Environment(name="myenv2")
conda_dependencies = CondaDependencies().create()
conda_dependencies.set_python_version("3.6.2")
conda_dependencies.add_conda_package("pip==20.2.4")
for pack in pip_packages:
    conda_dependencies.add_pip_package(pack)
env.python.conda_dependencies = conda_dependencies
env.register(workspace=ws)

# Run configuration with defined compute target and environment
aml_run_config = RunConfiguration()
aml_run_config.target = aml_compute_target
aml_run_config.node_count = 1
aml_run_config.environment = env
aml_run_config.environment_variables = environment_variables

cwd = os.getcwd()
# Arguments for pipeline step creation
kwargs_modelling = {
    "name": "run_modelling_pipeline",
    "script_name": "./src/concatenate_scripts/concatenate_scripts.py",
    "source_directory": "{cwd}".format(cwd=cwd),
    "compute_target": aml_compute_target,
    "runconfig": aml_run_config,
    "allow_reuse": False,
}
# Step object
run_modelling_pipeline = PythonScriptStep(**kwargs_modelling)

# Pipeline object
pipeline = Pipeline(
    workspace=ws,
    steps=[run_modelling_pipeline],
    default_source_directory="{cwd}".format(cwd=cwd),
)

# Publishing pipeline that will be moved to
# existing or new endpoint.
published_pipeline = pipeline.publish(name=f"test-modelling-pipeline", description="")


def update_endpoint(
    aml_workspace: Workspace,
    published_pipeline: PublishedPipeline,
    endpoint_name: str,
    description: str,
) -> None:
    try:
        pipeline_endpoint = PipelineEndpoint.get(
            workspace=aml_workspace, name=endpoint_name
        )
        pipeline_endpoint.add(published_pipeline)
        pipeline_endpoint.set_default(published_pipeline)
    except Exception as error:
        print(
            f"""
            {error} Endpoint was not found. Publishing a new one.
        """
        )
        pipeline_endpoint = PipelineEndpoint.publish(
            workspace=aml_workspace,
            name=endpoint_name,
            pipeline=published_pipeline,
            description=description,
        )
        pipeline_endpoint.add_default(published_pipeline)


update_endpoint(
    ws,
    published_pipeline=published_pipeline,
    endpoint_name="test-modelling-pipeline",
    description="",
)

# Run pipeline and wait for completion
# experiment = Experiment(ws, "test")
# pipeline_run_exp = experiment.submit(pipeline)
# pipeline_run_exp.wait_for_completion()

# Downloading logs obtained while running pipeline to show on github workflow
# Downloading file with scoring uri
# for pipeline_run in pipeline_run_exp.get_steps():
#    pipeline_run.download_file("user_logs/std_log.txt")
