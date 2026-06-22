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


def get_query_parameters(query_id, headers):
    """Return the set of parameter keys a saved query declares.

    Returns ``None`` if the query metadata can't be read (e.g. private query or
    a transient error), so callers can fall back to sending params unchanged.
    """
    url = f"{BASE_URL}/query/{query_id}"
    data = requests.get(url, headers=headers).json()
    declared = data.get("parameters")
    if declared is None:
        return None
    return {p["key"] for p in declared if "key" in p}


def select_params(query_id, params, headers, label=""):
    """Keep only the entries of ``params`` that ``query_id`` actually declares.

    Lets a single superset ``params`` dict be reused across every query: keys a
    given query does not declare (e.g. ``token0`` / ``token1`` for a chain-wide
    gas query) are dropped instead of triggering an "unknown parameters" error.
    Falls back to the full ``params`` if the declared set can't be determined.
    """
    declared = get_query_parameters(query_id, headers)
    if declared is None:
        return params

    selected = {k: v for k, v in params.items() if k in declared}
    dropped = sorted(set(params) - set(selected))
    if dropped:
        print(f"[{label}] dropping params not used by query {query_id}: {dropped}")
    return selected


def execute_query(query_id, params, headers, label=""):
    """Trigger a saved query and return its ``execution_id`` (or ``None``).

    Only the parameters the query declares are sent, so passing a superset (a
    shared ``params`` dict) is safe.
    """
    params = select_params(query_id, params, headers, label)

    url = f"{BASE_URL}/query/{query_id}/execute"
    payload = {"query_parameters": params}

    data = requests.post(url, json=payload, headers=headers).json()
    print(f"[{label}] EXECUTE RESPONSE:", data)

    if "error" in data:
        print(f"[WARNING] {label}: not available for chain '{params.get('chain')}'")
        print(f"Reason: {data['error']}")
        return None

    return data.get("execution_id")


def wait_for_execution(execution_id, headers, label="", sleep=3, max_wait=None):
    """Poll the execution status until it completes. Returns success flag.

    If ``max_wait`` (seconds) is given and exceeded before the query completes,
    give up and return ``False`` so the caller treats it like a failure (empty
    DataFrame) instead of blocking the notebook on a slow query.
    """
    url = f"{BASE_URL}/execution/{execution_id}/status"
    start = time.time()

    while True:
        state = requests.get(url, headers=headers).json().get("state")
        print(f"[{label}] STATUS:", state)

        if state == "QUERY_STATE_COMPLETED":
            return True
        if state == "QUERY_STATE_FAILED":
            print(f"[WARNING] {label}: query failed")
            return False
        if max_wait is not None and time.time() - start > max_wait:
            print(f"[WARNING] {label}: timed out after {max_wait}s")
            return False

        time.sleep(sleep)


def fetch_results(execution_id, headers, label=""):
    """Download the result rows of a completed execution as a ``DataFrame``.

    Dune sometimes returns space-padded column names and string values; they are
    stripped here so the saved CSVs are clean and merges on keys like
    ``evt_tx_hash`` / ``pool`` match across queries.
    """
    url = f"{BASE_URL}/execution/{execution_id}/results"

    rows = requests.get(url, headers=headers).json().get("result", {}).get("rows", [])
    df = pd.DataFrame(rows)

    if df.empty:
        print(f"[INFO] {label}: no data returned")
        return df

    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    return df


def run_dune_saved_query(query_id, params, headers, label="", max_wait=None):
    """Run a saved query end-to-end (execute -> wait -> fetch).

    Returns an empty ``DataFrame`` if the query is unavailable, fails, or does
    not finish within ``max_wait`` seconds, so the caller can keep going for the
    other DEXes.
    """
    execution_id = execute_query(query_id, params, headers, label)
    if execution_id is None:
        return pd.DataFrame()

    if not wait_for_execution(execution_id, headers, label, max_wait=max_wait):
        return pd.DataFrame()

    return fetch_results(execution_id, headers, label)
