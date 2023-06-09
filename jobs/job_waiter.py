"""_summary_
"""
import time
import requests
import json
import sys
import os

job_payload = {"job_id": int(sys.argv[1])}

resp = requests.get(
    f'{os.getenv("DATABRICKS_HOST")}/api/2.1/jobs/runs/list',
    json=job_payload,
    headers={"Authorization": f'Bearer {os.getenv("DATABRICKS_TOKEN")}'},
)

list_runs_actual_job = []
list_runs_actual_job_time = []
for elem in json.loads(resp.text)["runs"]:
    if elem["job_id"] == int(sys.argv[1]):
        list_runs_actual_job.append(elem["run_id"])
        list_runs_actual_job_time.append(elem["start_time"])

needed_index = list_runs_actual_job_time.index(max(list_runs_actual_job_time))
needed_run_id = list_runs_actual_job[needed_index]

state = "UNKNOWN"
while state not in ["FAILED", "SUCCESS"]:
    job_payload = {"run_id": needed_run_id}

    resp = requests.get(
        f'{os.getenv("DATABRICKS_HOST")}/api/2.1/jobs/runs/get',
        json=job_payload,
        headers={"Authorization": f'Bearer {os.getenv("DATABRICKS_TOKEN")}'},
    )

    try:
        state_ = json.loads(resp.text)["state"]["result_state"]
        if state_ not in ["FAILED", "SUCCESS"]:
            time.sleep(300)
        else:
            state = state_
    except KeyError:
        time.sleep(300)

print(f"state={state}")
