"""
"""
from azureml.pipeline.core import PipelineEndpoint
from azureml.core import Workspace, Experiment
from azureml.core.authentication import ServicePrincipalAuthentication
import os

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

pipeline_endpoint = PipelineEndpoint.get(workspace=ws, name="test-modelling-pipeline")
experiment = Experiment(ws, "test")
pipeline_run = experiment.submit(pipeline_endpoint)
pipeline_run.wait_for_completion()
