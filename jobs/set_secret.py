import requests
import os
import sys
from base64 import b64encode
from nacl import encoding, public
import json


def encrypt(public_key: str, secret_value: str):
    public_key = public.PublicKey(public_key.encode("utf-8"), encoding.Base64Encoder())
    sealed_box = public.SealedBox(public_key)
    encrypted = sealed_box.encrypt(secret_value.encode("utf-8"))
    return b64encode(encrypted).decode("utf-8")


resp = requests.get(
    "https://api.github.com/repos/lgriva-ext/ml_solution/actions/secrets/public-key",
    headers={"Authorization": f'Bearer {os.getenv("GITHUB_TOKEN")}'},
)

encrypted_value = encrypt(json.loads(resp.text)["key"], f"{sys.argv[1]}")
resp = requests.put(
    "https://api.github.com/repos/lgriva-ext/ml_solution/actions/secrets/JOB_ID_ACTIVE_PROD_JOB",
    json={
        "encrypted_value": encrypted_value,
        "key_id": json.loads(resp.text)["key_id"],
    },
    headers={"Authorization": f'Bearer {os.getenv("GITHUB_TOKEN")}'},
)

print(resp.status_code)
