"""_summary_
"""
import requests
import sys
import os


stage = sys.argv[3]
archived = False
if stage == "Production":
    archived = True
job_payload = {
    "name": sys.argv[1],
    "version": sys.argv[2],
    "stage": stage,
    "archive_existing_versions": archived,
}

resp = requests.post(
    f'{os.getenv("MODEL_REGISTRY_HOST")}/api/2.0/mlflow/transition-requests/approve',
    json=job_payload,
    headers={"Authorization": f'Bearer {os.getenv("MODEL_REGISTRY_TOKEN")}'},
)

print(resp.status_code)
