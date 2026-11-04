"""Refresh the Kaggle OAuth token and persist it where the CLI reads it.

Why this exists: Kaggle CLI 2.2.5 authenticates from ~/.kaggle/credentials.json (OAuth)
or from ~/.kaggle/access_token / $KAGGLE_API_TOKEN. The access token in
credentials.json is short-lived (~6h). The SDK can refresh it using the stored
refresh_token, but it does not persist the result, and the CLI does not refresh on its
own -> "Authentication required to call the Kaggle API" and 401s on every call.

After ONE interactive `kaggle auth login` (browser), the stored refresh_token is live
and this script keeps the session alive unattended from then on. If the refresh token
itself has been revoked or expired, no script can fix it - only a new browser login.
That is the difference between the two failures seen in one session: a dead
refresh_token (401 on every call, refresh returns the stale token) versus a live one.

    python3 scripts/kaggle_refresh.py && kaggle quota

Exit codes: 0 = refreshed and verified; 2 = the token was rejected (needs a browser
login); 3 = the refresh call itself failed (network/endpoint problem).
"""
import json
import os
import stat
import sys
from datetime import datetime, timedelta, timezone

CREDS = os.path.expanduser("~/.kaggle/credentials.json")
TOKEN_FILE = os.path.expanduser("~/.kaggle/access_token")


def verify(token: str) -> bool:
    """Prove the token works with a cheap authenticated call before persisting it."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.config_values["token"] = token
        api.benchmarks_quota()
        return True
    except Exception as e:                                   # noqa: BLE001
        print(f"verification failed: {type(e).__name__}: {str(e)[:160]}")
        return False


def main() -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    with api.build_kaggle_client() as client:
        from kagglesdk.kaggle_creds import KaggleCredentials
        creds = KaggleCredentials.load(client=client)
    if not creds:
        print(f"no credentials found at {CREDS}")
        return 2
    try:
        token = creds.get_access_token()          # refreshes if expired
    except Exception as e:                        # noqa: BLE001
        print(f"refresh failed: {type(e).__name__}: {str(e)[:160]}")
        print("if the refresh token was revoked, run: kaggle auth login --force")
        return 3
    if not token:
        print("refresh produced no token")
        return 2

    if not verify(token):
        print("the refreshed token was rejected by the API -> an interactive "
              "`kaggle auth login --force` is required")
        return 2

    with open(TOKEN_FILE, "w") as fh:
        fh.write(token)
    os.chmod(TOKEN_FILE, stat.S_IRUSR | stat.S_IWUSR)

    with open(CREDS) as fh:
        data = json.load(fh)
    data["access_token"] = token
    data["access_token_expiration"] = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
    with open(CREDS, "w") as fh:
        json.dump(data, fh, indent=2)
    os.chmod(CREDS, stat.S_IRUSR | stat.S_IWUSR)
    print("refreshed + verified kaggle token; expiry ~", data["access_token_expiration"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
