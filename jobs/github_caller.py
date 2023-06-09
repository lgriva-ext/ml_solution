# Databricks notebook source
"""_summary_
"""
import requests
import json


to_stage = json.loads(dbutils.widgets.get("event_message"))["to_stage"]
choosen_env = to_stage.lower()
model_name = json.loads(dbutils.widgets.get("event_message"))["model_name"]
model_version = json.loads(dbutils.widgets.get("event_message"))["version"]

if choosen_env == "staging":
    job_payload = {
        "ref": "main",
        "inputs": {
            "choosenEnv": choosen_env,
            "stageChangeRequested": "true",
            "modelName": model_name,
            "modelVersion": model_version,
            "toStage": to_stage,
        },
    }
    resp = requests.post(
        "https://api.github.com/repos/lgriva-ext/ml_solution/actions/workflows/cd_databricks_staging.yml/dispatches",
        json=job_payload,
        headers={
            "Authorization": f'Bearer {dbutils.secrets.get(scope="github", key="github-token")}'
        },
    )

    print(resp.status_code)

elif choosen_env == "production":
    pass
