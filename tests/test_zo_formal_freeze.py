import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from src.distributed.zo_formal import (
    DEFAULT_FREEZE,
    PACKAGED_FROZEN_INPUTS,
    load_and_verify_freeze,
    resolve_frozen_input,
)


def test_packaged_frozen_inputs_match_recorded_hashes():
    freeze = json.loads(DEFAULT_FREEZE.read_text(encoding="utf-8"))
    for relative, expected in freeze["input_sha256"].items():
        packaged = PACKAGED_FROZEN_INPUTS[relative]
        assert packaged.is_file()
        assert hashlib.sha256(packaged.read_bytes()).hexdigest() == expected


def test_resolver_uses_packaged_copy_when_original_is_absent(tmp_path: Path):
    relative = "outputs/test-only/missing-frozen-input.csv"
    packaged = tmp_path / "packaged.csv"
    packaged.write_text("value\n1\n", encoding="utf-8")
    with patch.dict(PACKAGED_FROZEN_INPUTS, {relative: packaged}):
        assert resolve_frozen_input(relative) == packaged


def test_repository_freeze_verifies_without_rewriting_audit_hashes():
    assert load_and_verify_freeze(DEFAULT_FREEZE)["status"] == "frozen"
