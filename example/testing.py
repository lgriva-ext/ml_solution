import argparse
from datetime import datetime
import pandas as pd
import pytz

try:
    import azureml.core
    from azureml.core import Dataset, Model
    from azureml.core.run import Run, _OfflineRun
    from azureml.core import Workspace, Webservice
except:
    pass
import urllib.request
import json
import os
import ssl

try:
    ws = Workspace.from_config()
except:
    pass


def allowSelfSignedHttps(allowed):
    if (
        allowed
        and not os.environ.get("PYTHONHTTPSVERIFY", "")
        and getattr(ssl, "_create_unverified_context", None)
    ):
        ssl._create_default_https_context = ssl._create_unverified_context


allowSelfSignedHttps(True)

parser = argparse.ArgumentParser("split")
parser.add_argument("--path_to_test", help="Path of testing data")
args = parser.parse_args()

json_data = json.load(open(args.path_to_test, "r"))
df1 = pd.DataFrame(json_data, index=[0])

test_sample = json.dumps(
    {
        "data": json.loads(df1.to_json(orient="records")),
    }
)
body = str.encode(test_sample)

url = "http://9d6d16ee-8434-4969-8a40-49e6b8f14f72.eastus2.azurecontainer.io/score"

try:
    api_key = ""  # Replace this with the API key for the web service
    headers = {
        "Content-Type": "application/json",
        "Authorization": ("Bearer " + api_key),
    }
    req = urllib.request.Request(url, body, headers)
    response = urllib.request.urlopen(req)
    result = response.read()
    print("Llamando a la url del servicio")
    print(
        "La predicción para las features provistas es: {}".format(eval(result)[12:19])
    )
except:
    print(req)

try:
    service = Webservice(ws, "bigmart-reg-svc")
    print("Llamando al objeto Webservice")
    print(
        "La predicción para las features provistas es: {}".format(
            eval(service.run(input_data=test_sample))["predict"]
        )
    )
except:
    pass
