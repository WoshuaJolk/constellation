"""Local credential profiles. No secret is included in requests saved as receipts."""
import json
import os
import stat
from pathlib import Path
from urllib.parse import quote, quote_plus

DEFAULT_PATH = Path.home() / ".config" / "magictools" / "credentials.json"
ENV = {"hsc": {"username": "HSC_SSP_CAS_USER", "password": "HSC_SSP_CAS_PASSWORD"},
       "mast": {"token": "MAST_API_TOKEN"},
       "noirlab": {"token": "NOIRLAB_API_TOKEN"},
       "irsa": {"username": "IRSA_USER", "password": "IRSA_PASSWORD"}}


class Credentials:
    def __init__(self, path=None):
        self.profiles = {}
        self.secrets = []
        path = Path(path) if path else DEFAULT_PATH
        if path.exists():
            if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) & 0o077:
                raise ValueError(f"Credential file must be owner-only: chmod 600 {path}")
            data = json.loads(path.read_text())
            if not isinstance(data, dict) or not isinstance(data.get("providers", {}), dict):
                raise ValueError("Credential config must contain a providers object")
            self.profiles = data.get("providers", {})
            if any(not isinstance(profile, dict) for profile in self.profiles.values()):
                raise ValueError("Each credential profile must be an object")

    def get(self, provider):
        result = {}
        profile = self.profiles.get(provider, {})
        for key in {"username", "password", "token"}:
            entry = profile.get(key)
            env = ENV.get(provider, {}).get(key)
            if isinstance(entry, dict):
                if "env" in entry:
                    result[key] = os.environ.get(entry["env"], "")
                elif "file" in entry:
                    path = Path(entry["file"]).expanduser()
                    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) & 0o077:
                        raise ValueError("Credential secret file must be owner-only")
                    result[key] = path.read_text().strip()
            elif isinstance(entry, str):
                result[key] = entry
            elif env:
                result[key] = os.environ.get(env, "")
        self.secrets.extend(v for k, v in result.items() if k in {"password", "token"} and v and v not in self.secrets)
        return result

    def redact(self, text):
        for secret in self.secrets:
            for encoded in (secret, json.dumps(secret)[1:-1], quote(secret, safe=""), quote_plus(secret)):
                text = text.replace(encoded, "[REDACTED]")
        return text

    def status(self):
        output = {}
        for name, fields in ENV.items():
            values = self.get(name)
            output[name] = {"configured": bool(values.get("token") or (values.get("username") and values.get("password"))),
                            "environmentVariables": fields, "validatedWithArchive": False}
        return output


def initialize(path=None):
    path = Path(path) if path else DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"providers": {name: {k: {"env": v} for k, v in fields.items()} for name, fields in ENV.items()}}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(data, stream, indent=2)
    return {"path": str(path), "created": True, "containsSecrets": False}
