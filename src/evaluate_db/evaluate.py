# Databricks notebook source
"""_summary_
"""
import requests
import os
import mlflow
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
import logging
import json


to_stage = json.loads(dbutils.widgets.get("event_message"))["to_stage"]
choosen_env = to_stage.lower()
model_name = json.loads(dbutils.widgets.get("event_message"))["model_name"]
model_version = json.loads(dbutils.widgets.get("event_message"))["version"]


def compliance_checks_approved(model_name, model_version):
    """_summary_

    Args:
        model_name (_type_): _description_
        model_version (_type_): _description_
    """
    # DOWNLOAD MODEL ARTIFACTS
    # download_model_artifacts(model_name, model_version)
    # logged_model = f"models:/{model_name}/{model_version}"
    # loaded_model = mlflow.pyfunc.load_model(logged_model)
    # run_id = loaded_model.metadata.run_id
    # CHECK TAGS, METRICS, AMOUNT OF ARTIFACTS, FLAVOR OF MODEL AND SO ON
    # metrics = mlflow.get_run(run_id=run_id).to_dictionary()["data"]["metrics"]
    return True


def download_model_artifacts():
    mlflow.set_tracking_uri("databricks")

    os.makedirs("model", exist_ok=True)
    local_path = ModelsArtifactRepository(
        f'models:/{model_name}/{model_version}').download_artifacts("", dst_path="model")


def promote_to_staging(name, version):
    job_payload = {
        "name": name,
        "version": version,
        "stage": "Staging",
        "archive_existing_versions": False,
        "comment": "Staging version of this model",
    }

    resp = requests.post(
        f'{dbutils.secrets.get(scope="modelregistery", key="modelregistery-host")}api/2.0/mlflow/transition-requests/approve',
        json=job_payload,
        headers={
            "Authorization": f'Bearer {dbutils.secrets.get(scope="modelregistery", key="modelregistery-token")}'
        },
    )

    print(resp.status_code)


def load_current_production_model(model_name):
    job_payload = {"filter": f"name='{model_name}'"}

    resp = requests.get(
        f'{dbutils.secrets.get(scope="modelregistery", key="modelregistery-host")}api/2.0/mlflow/model-versions/search',
        json=job_payload,
        headers={
            "Authorization": f'Bearer {dbutils.secrets.get(scope="modelregistery", key="modelregistery-token")}'
        },
    )

    list_model_in_prod_versions = []
    list_model_in_prod_cretion = []
    for elem in json.loads(resp.text)["model_versions"]:
        if elem["current_stage"] == "Production":
            list_model_in_prod_versions.append(elem["version"])
            # list_model_in_staging_cretion.append(elem["last_updated_timestamp"])
            list_model_in_prod_cretion.append(elem["creation_timestamp"])

    if list_model_in_prod_cretion != []:
        index_last_model = list_model_in_prod_cretion.index(
            max(list_model_in_prod_cretion)
        )
        last_model_version = list_model_in_prod_versions[index_last_model]

        return last_model_version
    else:
        return "x"


def test_against_current_production(model_name, model_version):
    model = load_current_production_model(model_name)
    if model == "x":
        print("First production model trained")
    return True


def request_to_prod_transition(name, version):
    job_payload = {
        "name": name,
        "version": version,
        "stage": "Production",
        "comment": "Production version of this model",
    }

    resp = requests.post(
        f'{dbutils.secrets.get(scope="modelregistery", key="modelregistery-host")}api/2.0/mlflow/transition-requests/create',
        json=job_payload,
        headers={
            "Authorization": f'Bearer {dbutils.secrets.get(scope="modelregistery", key="modelregistery-token")}'
        },
    )

    print(resp.status_code)


def archived_model(name, version):
    job_payload = {
        "name": name,
        "version": version,
        "stage": "Archived",
        "comment": "Staging version of this model",
    }

    resp = requests.post(
        f'{dbutils.secrets.get(scope="modelregistery", key="modelregistery-host")}api/2.0/mlflow/databricks/model-versions/transition-stage',
        json=job_payload,
        headers={
            "Authorization": f'Bearer {dbutils.secrets.get(scope="modelregistery", key="modelregistery-token")}'
        },
    )

    print(resp.status_code)


if __name__ == "__main__":
    if choosen_env == "staging":
        # Run compliance checks before approve "to Staging" transition
        print("Running compliance checks")
        if compliance_checks_approved(model_name, model_version):
            print(f"Promoting model {model_name}, version {model_version} to Staging")
            promote_to_staging(model_name, model_version)
            print("Test registered model against last production registered model")
            to_prod_stage = test_against_current_production(model_name, model_version)
            if to_prod_stage:
                request_to_prod_transition(model_name, model_version)
            else:
                print(
                    "Model accuracy is not better than current production model accuracy, it will be archived"
                )
                archived_model(model_name, model_version)
        else:
            print("Model did not pass compliance checks, it will be archived")
            archived_model(model_name, model_version)
