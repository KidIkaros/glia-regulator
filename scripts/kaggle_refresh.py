"""Refresh the Kaggle OAuth token and persist it where the CLI looks.

Why: Kaggle CLI 2.2.5 authenticates from ~/.kaggle/credentials.json (OAuth) or from
~/.kaggle/access_token / $KAGGLE_API_TOKEN. The SDK can refresh the expired access
token using the stored refresh_token, but it does not persist the result, and the CLI
does not refresh on its own -> "Authentication required to call the Kaggle API".

This script refreshes, writes ~/.kaggle/access_token (0600), and updates
credentials.json in place. Run it before any CLI call that matters:

    python3 scripts/kaggle_refresh.py && kaggle quota
"""
import json
import os
import stat
import sys
from datetime import datetime, timedelta, timezone

CREDS = os.path.expanduser("~/.kaggle/credentials.json")
TOKEN_FILE = os.path.expanduser("~/.kaggle/access_token")


def main() -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    creds = None
    with api.build_kaggle_client() as client:
        from kagglesdk.kaggle_creds import KaggleCredentials
        creds = KaggleCredentials.load(client=client)
    if not creds:
        print("no credentials found at", CREDS)
        return 1
    token = creds.get_access_token()          # refreshes if expired
    if not token:
        print("refresh produced no token")
        return 1

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
    print("refreshed kaggle token; expiry ~", data["access_token_expiration"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
