# Databricks notebook source
"""_summary_
"""
import requests
import logging
import json
import os


to_stage = json.loads(dbutils.widgets.get("event_message"))["to_stage"]
choosen_env = to_stage.lower()
model_name = json.loads(dbutils.widgets.get("event_message"))["model_name"]
model_version = json.loads(dbutils.widgets.get("event_message"))["version"]


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
    if choosen_env == "production":
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
            print("One or more models in production founded")

            for ix, elem in enumerate(list_model_in_prod_versions):
                # if ix != index_last_model:
                archived_model(model_name, elem)

        # promote_to_production(model_name, model_version)
