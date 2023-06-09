"""_summary_
"""
import requests
import sys
import os


job_payload = {
    "name": sys.argv[1],
    "version": sys.argv[2],
    "stage": "None",
    "archive_existing_versions": False,
    "comment": "Could not pass Staging environment tests",
}

resp = requests.post(
    f'{os.getenv("MODEL_REGISTRY_HOST")}/api/2.0/mlflow/databricks/model-versions/transition-stage',
    json=job_payload,
    headers={"Authorization": f'Bearer {os.getenv("MODEL_REGISTRY_TOKEN")}'},
)

print(resp.status_code)
