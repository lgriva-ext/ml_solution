"""_summary_
"""
import requests
import sys
import os


job_payload = {
    "job_id": sys.argv[1],
    "new_settings": {"schedule": {"pause_status": "PAUSED", "quartz_cron_expression": "36 26 10 * * ?", "timezone_id": "America/Argentina/Buenos_Aires"}}
    }

resp = requests.post(
    f'{os.getenv("DATABRICKS_HOST")}/api/2.1/jobs/update',
    json=job_payload,
    headers={"Authorization": f'Bearer {os.getenv("DATABRICKS_TOKEN")}'},
)

print(resp.status_code)
