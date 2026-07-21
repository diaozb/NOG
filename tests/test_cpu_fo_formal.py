import copy
import json
import unittest
from pathlib import Path

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.cpu_fo_formal import (
    unique_formal_configs,
    validate_formal_manifest,
    validate_frozen_selection,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "distributed_cpu_fo_pilot.yaml"
PILOT_ROOT = ROOT / "outputs" / "distributed_cpu_fo" / "pilot"


class CpuFoFormalTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(CONFIG)
        with open(
            PILOT_ROOT / "selected_config_by_epsilon.yaml",
            "r",
            encoding="utf-8",
        ) as handle:
            self.frozen = json.load(handle)
        with open(
            PILOT_ROOT / "pilot_final_report.json",
            "r",
            encoding="utf-8",
        ) as handle:
            self.report = json.load(handle)

    def test_frozen_selection_and_unique_config_deduplication(self):
        validate_frozen_selection(self.cfg, self.frozen, self.report)
        configs = unique_formal_configs(self.cfg, self.frozen)
        self.assertEqual(len(configs), 6)
        self.assertEqual(sum(row["method"] == "NOG-FO" for row in configs), 3)
        self.assertEqual(sum(row["method"] == "ME-DOL-FO" for row in configs), 3)
        covered = sorted(
            (row["method"], epsilon)
            for row in configs
            for epsilon in row["epsilons"]
        )
        self.assertEqual(len(covered), 10)

    def test_frozen_hash_tampering_is_rejected(self):
        tampered = copy.deepcopy(self.frozen)
        tampered["by_epsilon"]["0.01"]["NOG-FO"]["selected_parameters"][
            "eta"
        ] = 999.0
        with self.assertRaisesRegex(ValueError, "SHA256"):
            validate_frozen_selection(self.cfg, tampered, self.report)

    def test_formal_manifest_task_tampering_is_rejected(self):
        with open(
            ROOT
            / "outputs"
            / "distributed_cpu_fo"
            / "formal_accuracy"
            / "formal_task_manifest.json",
            "r",
            encoding="utf-8",
        ) as handle:
            manifest = json.load(handle)
        validate_formal_manifest(self.cfg, manifest, self.frozen, self.report)
        tampered = copy.deepcopy(manifest)
        tampered["formal_configs"][0]["tasks"][0]["formal_seed"] = 100
        with self.assertRaisesRegex(ValueError, "task identities"):
            validate_formal_manifest(self.cfg, tampered, self.frozen, self.report)


if __name__ == "__main__":
    unittest.main()
