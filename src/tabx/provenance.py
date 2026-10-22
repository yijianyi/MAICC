"""Content identities for training artifacts; paths alone are not identities."""
import hashlib
import json
import subprocess
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity():
    root = Path(__file__).resolve().parents[2]
    files = sorted(p for folder in ("src", "tools")
                   for p in (root / folder).rglob("*")
                   if p.suffix in (".py", ".yaml", ".sh"))
    hashes = {str(p.relative_to(root)): file_sha256(p) for p in files}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git_commit": commit, "source_sha256": hashes}


def encoder_sha256(state):
    """Same identity before/after DT-only training, independent of ZIP metadata."""
    digest = hashlib.sha256()
    for name in sorted(state):
        if name.startswith("DT.") or name.startswith("DT_"):
            continue
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def write_training_provenance(directory, config, manifest, agent):
    payload = source_identity()
    payload.update({
        "schema": "tabx_baseline_provenance_v2",
        "encoder_sha256": encoder_sha256(agent.state_dict()),
        "data_manifest_sha256": file_sha256(manifest.path),
        "normalization_sha256": (file_sha256(config["data"]["normalization"])
                                 if config["data"].get("normalization") else None),
        "cem_architecture": config.get("model", {}).get("cem_architecture", "dense"),
        "retrieval_protocol": "maicc_local_exact_prefix_v4",
        "context_length": manifest.sequence_length,
    })
    (Path(directory) / "provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
