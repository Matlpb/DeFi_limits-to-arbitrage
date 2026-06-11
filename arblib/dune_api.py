"""
Thin wrapper around the Dune Analytics REST API.

The flow for a saved query is always the same:

    execute  ->  poll status until COMPLETED  ->  fetch result rows

:func:`run_dune_saved_query` chains those three steps and returns a tidy
``DataFrame`` (empty if the query is not available for the requested chain or
if it fails).
"""

import time

import pandas as pd
import requests

BASE_URL = "https://api.dune.com/api/v1"


def make_headers(api_key):
    """Build the auth headers used by every request."""
    return {
        "X-Dune-API-Key": api_key,
        "Content-Type": "application/json",
    }


def execute_query(query_id, params, headers, label=""):
    """Trigger a saved query and return its ``execution_id`` (or ``None``)."""
    url = f"{BASE_URL}/query/{query_id}/execute"
    payload = {"query_parameters": params}

    data = requests.post(url, json=payload, headers=headers).json()
    print(f"[{label}] EXECUTE RESPONSE:", data)

    if "error" in data:
        print(f"[WARNING] {label}: not available for chain '{params.get('chain')}'")
        print(f"Reason: {data['error']}")
        return None

    return data.get("execution_id")


def wait_for_execution(execution_id, headers, label="", sleep=3):
    """Poll the execution status until it completes. Returns success flag."""
    url = f"{BASE_URL}/execution/{execution_id}/status"

    while True:
        state = requests.get(url, headers=headers).json().get("state")
        print(f"[{label}] STATUS:", state)

        if state == "QUERY_STATE_COMPLETED":
            return True
        if state == "QUERY_STATE_FAILED":
            print(f"[WARNING] {label}: query failed")
            return False

        time.sleep(sleep)


def fetch_results(execution_id, headers, label=""):
    """Download the result rows of a completed execution as a ``DataFrame``."""
    url = f"{BASE_URL}/execution/{execution_id}/results"

    rows = requests.get(url, headers=headers).json().get("result", {}).get("rows", [])
    df = pd.DataFrame(rows)

    if df.empty:
        print(f"[INFO] {label}: no data returned")

    return df


def run_dune_saved_query(query_id, params, headers, label=""):
    """Run a saved query end-to-end (execute -> wait -> fetch).

    Returns an empty ``DataFrame`` if the query is unavailable or fails so the
    caller can keep going for the other DEXes.
    """
    execution_id = execute_query(query_id, params, headers, label)
    if execution_id is None:
        return pd.DataFrame()

    if not wait_for_execution(execution_id, headers, label):
        return pd.DataFrame()

    return fetch_results(execution_id, headers, label)
