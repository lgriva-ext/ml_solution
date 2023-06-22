import sys
import os
import requests
import json


name = sys.argv[1]


def update_ep(name, version):
    aux_path = os.getcwd()
    config_path = f"{aux_path}/jobs/current_model.json"
    endpoint_name = json.load(open(config_path, "r"))["current_endpoint_name"]
    config = {
        "served_models": [
            {
                "model_name": f"{name}",
                "model_version": f"{version}",
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
            }
        ]
    }
    data = {"name": f"{endpoint_name}", "config": config}

    headers = {
        "Authorization": f'Bearer {os.getenv("PROD_ENV_TOKEN")}',
        "Content-Type": "application/json",
    }
    resp = requests.post(
        f"{os.getenv('PROD_ENV_HOST')}api/2.0/serving-endpoints",
        json=data,
        headers=headers,
    )

    print(resp.status_code)
    print(resp.text)

    if resp.status_code == 400:
        if json.loads(resp.text)["error_code"] == "RESOURCE_ALREADY_EXISTS":
            resp = requests.put(
                f"{os.getenv('PROD_ENV_HOST')}api/2.0/serving-endpoints/{endpoint_name}/config",
                json=config,
                headers=headers,
            )

    print(resp.status_code)
    print(resp.text)


def delete_current_production_model(version):

    job_payload = {
        "name": name,
        "version": version,
    }

    resp = requests.delete(
        f'{os.getenv("MODEL_REGISTRY_HOST")}api/2.0/mlflow/model-versions/delete',
        json=job_payload,
        headers={
            "Authorization": f'Bearer {os.getenv("MODEL_REGISTRY_TOKEN")}'
        },
    )

    print(resp.status_code)
    print(resp.text)


def get_previous_prod_model():
    job_payload = {
        "name": name,
        "stage": "Production",
    }

    resp = requests.post(
        f'{os.getenv("MODEL_REGISTRY_HOST")}api/2.0/mlflow/registered-models/get-latest-versions',
        json=job_payload,
        headers={
            "Authorization": f'Bearer {os.getenv("MODEL_REGISTRY_TOKEN")}'
        },
    )

    version_older = json.loads(resp.text)["model_versions"][0]["version"]

    job_payload = {
        "name": name,
    }

    resp = requests.post(
        f'{os.getenv("MODEL_REGISTRY_HOST")}api/2.0/mlflow/registered-models/get-latest-versions',
        json=job_payload,
        headers={
            "Authorization": f'Bearer {os.getenv("MODEL_REGISTRY_TOKEN")}'
        },
    )

    json_resp = json.loads(resp.text)

    for elem in json_resp["model_versions"]:
        if elem["current_stage"] == "Production":
            version_new = elem["version"]

    return version_older, version_new


def promote_to_prod_last_prod_model(version):
    job_payload = {
        "name": name,
        "version": version,
        "stage": "Production",
        "archive_existing_versions": False,
        "comment": "Staging version of this model",
    }

    resp = requests.post(
        f'{os.getenv("MODEL_REGISTRY_HOST")}api/2.0/mlflow/databricks/model-versions/transition-stage',
        json=job_payload,
        headers={
            "Authorization": f'Bearer {os.getenv("MODEL_REGISTRY_TOKEN")}'
        },
    )

    print(resp.status_code)


def archive_current_production_model(version):

    job_payload = {
        "name": name,
        "version": version,
        "stage": "Archived",
        "archive_existing_versions": False,
        "comment": "Archiving version of this model",
    }

    resp = requests.post(
        f'{os.getenv("MODEL_REGISTRY_HOST")}api/2.0/mlflow/databricks/model-versions/transition-stage',
        json=job_payload,
        headers={
            "Authorization": f'Bearer {os.getenv("MODEL_REGISTRY_TOKEN")}'
        },
    )
    print(resp.status_code)


if __name__ == "__main__":
    version_prev_prod, version_curr_prod = get_previous_prod_model()
    archive_current_production_model(version_curr_prod)
    delete_current_production_model(version_curr_prod)
    promote_to_prod_last_prod_model(version_prev_prod)
    update_ep(name, version_prev_prod)
