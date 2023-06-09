"""_summary_
"""
import requests
import sys
import os
import json


job_payload = {"filter": f"name='{sys.argv[1]}'"}

resp = requests.get(
    f'{os.getenv("MODEL_REGISTRY_HOST")}/api/2.0/mlflow/model-versions/search',
    json=job_payload,
    headers={"Authorization": f'Bearer {os.getenv("MODEL_REGISTRY_TOKEN")}'},
)

list_model_in_staging_versions = []
list_model_in_staging_cretion = []
for elem in json.loads(resp.text)["model_versions"]:
    if elem["current_stage"] == "Staging":
        list_model_in_staging_versions.append(elem["version"])
        # list_model_in_staging_cretion.append(elem["last_updated_timestamp"])
        list_model_in_staging_cretion.append(elem["creation_timestamp"])

index_last_model = list_model_in_staging_cretion.index(
    max(list_model_in_staging_cretion)
)
last_model_version = list_model_in_staging_versions[index_last_model]

job_payload = {
    "name": sys.argv[1],
    "version": last_model_version,
    "stage": "Production",
}

resp = requests.post(
    f'{os.getenv("MODEL_REGISTRY_HOST")}/api/2.0/mlflow/transition-requests/create',
    json=job_payload,
    headers={"Authorization": f'Bearer {os.getenv("MODEL_REGISTRY_TOKEN")}'},
)

print(resp.status_code)
