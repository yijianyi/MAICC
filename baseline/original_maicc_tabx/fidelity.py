from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict


UPSTREAM_COMMIT = "e1640a9"

# SHA-256 over UTF-8 text with CRLF normalized to LF. These are the files from
# LAMDA-RL/MAICC at UPSTREAM_COMMIT.
EXPECTED_CORE_HASHES: Dict[str, str] = {
    "src/main.py": "e0005b7823ca3bd4913e4405b61f8c975248971a82f5a106fc5620896646447c",
    "src/run.py": "04a6f7262448f0ba147e399062fb77542512286e83d2b544809f4fa08ed3a99c",
    "src/config/algs/MAICC.yaml": "3f0910c58449f5055c5b519eb10f63c0f52184d5a3f44477d170992da7da6b7c",
    "src/controllers/ic_controller.py": "428e61bdac6c2eac91a20b304ffa48615becbcf0150283c86f36e66352836708",
    "src/modules/agents/transformer_agent.py": "4686e374dcefab094c79d3aa3f69c147825e4dbfd6ee9da47925719223b79eaa",
    "src/components/episode_buffer.py": "7d29549fcf7221f87915030d746a42330b358bbc9df870ff5a69744a16066fca",
}

EXPECTED_VENDOR_LEARNER_HASH = (
    "812e4f3fc934d312818384395504cc14d6a59911f811dfa00d072bf30e3a9381"
)


class FidelityError(RuntimeError):
    pass


def normalized_text_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_upstream_core(repo_root: Path) -> None:
    repo_root = Path(repo_root).resolve()
    mismatches = []
    for relative, expected in EXPECTED_CORE_HASHES.items():
        path = repo_root / relative
        if not path.is_file():
            mismatches.append(f"{relative}: missing")
            continue
        actual = normalized_text_hash(path)
        if actual != expected:
            mismatches.append(f"{relative}: {actual} != {expected}")
    vendor = (
        repo_root
        / "baseline"
        / "original_maicc_tabx"
        / "vendor"
        / "ic_learner.py"
    )
    if not vendor.is_file():
        mismatches.append("frozen original ICLearner: missing")
    else:
        actual = normalized_text_hash(vendor)
        if actual != EXPECTED_VENDOR_LEARNER_HASH:
            mismatches.append(
                "frozen original ICLearner: "
                f"{actual} != {EXPECTED_VENDOR_LEARNER_HASH}"
            )
    if mismatches:
        raise FidelityError(
            "Original MAICC fidelity check failed:\n  - "
            + "\n  - ".join(mismatches)
        )
