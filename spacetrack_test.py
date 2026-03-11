import os
import time

import requests


USERNAME = os.environ.get("SPACETRACK_USER", "")
PASSWORD = os.environ.get("SPACETRACK_PASS", "")


session = requests.Session()


# Login
login = session.post(
    "https://www.space-track.org/ajaxauth/login",
    data={"identity": USERNAME, "password": PASSWORD},
    timeout=30,
)
print(f"Login status: {login.status_code}")
print(f"Login response: {login.text[:200]}")
print(f"Cookies after login: {dict(session.cookies)}")


# Test query using same session
time.sleep(3)


resp = session.get(
    "https://www.space-track.org/basicspacedata/query/class/cdm_public/limit/2/format/json",
    timeout=30,
)
print(f"Query status: {resp.status_code}")
print(f"Query response: {resp.text[:300]}")
