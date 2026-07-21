import copy
import tempfile
import unittest
from pathlib import Path

from src.distributed.common import evaluation_seed, make_seed_bundle
from src.distributed.cpu_fo_correctness import load_config
from src.distributed.cpu_fo_pilot import (
    advancement_candidates,
    coarse_candidates,
    confirmed_hit,
    pareto_frontier,
    prepare_nog_batch_refinement,
    nog_refinement_work_audit,
    prepare_manifest,
    select_for_epsilon,
    unresolved_advancement_candidates,
    validate_3840_advancement_manifest,
    validate_nog_refinement_manifest,
    validate_advancement_manifest,
    validate_pilot_config,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "distributed_cpu_fo_pilot.yaml"


class CpuFoPilotTests(unittest.TestCase):
    def test_fixed_evaluation_bank_is_checkpoint_independent(self):
        seed = make_seed_bundle(100, "NOG-FO", 8)
        self.assertEqual(
            evaluation_seed(seed, 48, "fixed_bank"),
            evaluation_seed(seed, 960, "fixed_bank"),
        )
        self.assertNotEqual(
            evaluation_seed(seed, 48, "checkpoint"),
            evaluation_seed(seed, 960, "checkpoint"),
        )

    def test_grid_manifest_and_seed_separation(self):
        cfg = load_config(CONFIG)
        validate_pilot_config(cfg)
        candidates = coarse_candidates(cfg)
        self.assertEqual(len(candidates), 32)
        self.assertEqual(sum(row["method"] == "NOG-FO" for row in candidates), 20)
        self.assertEqual(sum(row["method"] == "ME-DOL-FO" for row in candidates), 12)
        with tempfile.TemporaryDirectory() as directory:
            manifest = prepare_manifest(cfg, directory)
            self.assertEqual(manifest["coarse_task_count"], 96)
            self.assertEqual(manifest["launches_started"], 0)
            self.assertTrue((Path(directory) / "candidate_manifest.json").exists())

        invalid = copy.deepcopy(cfg)
        invalid["pilot"]["seeds"] = [4, 100]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_pilot_config(invalid)

    def test_confirmed_hit_rejects_transient_crossing(self):
        values = [0.02, 0.009, 0.012, 0.008, 0.007]
        rows = [
            {
                "iteration": (index + 1) * 48,
                "stat_proxy": value,
                "depth": (index + 1) * 48,
                "total_work": (index + 1) * 100,
                "per_worker_work_max": (index + 1) * 10,
                "training_time": float(index + 1),
            }
            for index, value in enumerate(values)
        ]
        hit = confirmed_hit(rows, epsilon=0.01, consecutive=2)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit["iteration"], 192)
        self.assertEqual(hit["total_work"], 400)

    def test_pareto_and_selection(self):
        rows = [
            {
                "candidate_id": "fast",
                "hit_rate": 1.0,
                "full_hit": True,
                "median_depth": 10,
                "median_total_work": 100,
                "median_training_time": 5,
                "median_final_stat_proxy": 0.005,
            },
            {
                "candidate_id": "dominated",
                "hit_rate": 1.0,
                "full_hit": True,
                "median_depth": 12,
                "median_total_work": 120,
                "median_training_time": 6,
                "median_final_stat_proxy": 0.004,
            },
            {
                "candidate_id": "low-work",
                "hit_rate": 1.0,
                "full_hit": True,
                "median_depth": 20,
                "median_total_work": 50,
                "median_training_time": 8,
                "median_final_stat_proxy": 0.006,
            },
        ]
        frontier = pareto_frontier(rows)
        self.assertEqual({row["candidate_id"] for row in frontier}, {"fast", "low-work"})
        selected = select_for_epsilon(rows)
        self.assertEqual(selected["selected_candidate_id"], "fast")
        self.assertEqual(selected["min_work_candidate_id"], "low-work")

    def test_advancement_uses_frontier_or_top_censored(self):
        full_rows = [
            {
                "candidate_id": "frontier",
                "hit_rate": 1.0,
                "full_hit": True,
                "median_depth": 10,
                "median_total_work": 100,
                "median_training_time": 5,
                "median_final_stat_proxy": 0.005,
            },
            {
                "candidate_id": "dominated",
                "hit_rate": 1.0,
                "full_hit": True,
                "median_depth": 20,
                "median_total_work": 200,
                "median_training_time": 10,
                "median_final_stat_proxy": 0.004,
            },
        ]
        censored_rows = [
            {
                "candidate_id": identifier,
                "hit_rate": hit_rate,
                "full_hit": False,
                "median_depth": float("inf"),
                "median_total_work": float("inf"),
                "median_training_time": float("inf"),
                "median_final_stat_proxy": final,
            }
            for identifier, hit_rate, final in [
                ("best", 2 / 3, 0.009),
                ("second", 1 / 3, 0.008),
                ("excluded", 0.0, 0.02),
            ]
        ]
        advancing, reasons = advancement_candidates(
            {
                ("NOG-FO", 0.01): full_rows,
                ("NOG-FO", 0.008): censored_rows,
            },
            top_n_censored=2,
        )
        self.assertEqual(advancing, {"frontier", "best", "second"})
        self.assertNotIn("dominated", reasons)

    def test_advancement_manifest_rejects_parameter_tampering(self):
        cfg = load_config(CONFIG)
        candidate = coarse_candidates(cfg)[0]
        manifest = {
            "status": "prepared",
            "source_rounds": 960,
            "target_rounds": 1920,
            "launches_started": 0,
            "candidate_count": 1,
            "candidates": [copy.deepcopy(candidate)],
        }
        validated = validate_advancement_manifest(cfg, manifest)
        self.assertEqual(validated[0]["candidate_id"], candidate["candidate_id"])
        manifest["candidates"][0]["parameters"]["eta"] = 999.0
        with self.assertRaisesRegex(ValueError, "Parameter mismatch"):
            validate_advancement_manifest(cfg, manifest)

    def test_final_extension_advances_only_unresolved_thresholds(self):
        full = [
            {
                "candidate_id": "already-hit",
                "hit_rate": 1.0,
                "full_hit": True,
                "median_depth": 10,
                "median_total_work": 100,
                "median_training_time": 5,
                "median_final_stat_proxy": 0.005,
            }
        ]
        censored = [
            {
                "candidate_id": identifier,
                "hit_rate": hit_rate,
                "full_hit": False,
                "median_depth": depth,
                "median_total_work": work,
                "median_training_time": time,
                "median_final_stat_proxy": final,
            }
            for identifier, hit_rate, depth, work, time, final in [
                ("partial", 1 / 3, 50, 500, 5, 0.0076),
                ("best-zero", 0.0, float("inf"), float("inf"), float("inf"), 0.0077),
                ("excluded", 0.0, float("inf"), float("inf"), float("inf"), 0.009),
            ]
        ]
        advancing, _ = unresolved_advancement_candidates(
            {
                ("NOG-FO", 0.008): full,
                ("ME-DOL-FO", 0.0075): censored,
            },
            top_n_censored=2,
        )
        self.assertEqual(advancing, {"partial", "best-zero"})

    def test_3840_manifest_rejects_candidate_outside_prior_gate(self):
        cfg = load_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pilot"
            root.mkdir(parents=True)
            cfg["run"]["out_dir"] = str(root.parent)
            cfg["run"]["name"] = root.name
            candidates = coarse_candidates(cfg)
            prior = candidates[0]
            atomic_manifest = {
                "candidates": [prior],
            }
            from src.distributed.cpu_fo_tasks import atomic_write_json

            atomic_write_json(root / "advancement_to_1920.json", atomic_manifest)
            manifest = {
                "status": "prepared",
                "source_rounds": 1920,
                "target_rounds": 3840,
                "launches_started": 0,
                "candidate_count": 1,
                "candidates": [copy.deepcopy(prior)],
            }
            validated = validate_3840_advancement_manifest(cfg, manifest)
            self.assertEqual(len(validated), 1)
            manifest["candidates"] = [copy.deepcopy(candidates[1])]
            with self.assertRaisesRegex(ValueError, "did not pass"):
                validate_3840_advancement_manifest(cfg, manifest)

    def test_nog_refinement_reuses_coarse_batch(self):
        cfg = load_config(CONFIG)
        candidates = [
            row for row in coarse_candidates(cfg) if row["method"] == "NOG-FO"
        ][:2]
        rows = []
        for index, candidate in enumerate(candidates):
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "hit_rate": 1.0,
                    "full_hit": True,
                    "median_depth": 10 + index,
                    "median_total_work": 100 + index,
                    "median_training_time": 5 + index,
                    "median_final_stat_proxy": 0.005,
                }
            )
        manifest = prepare_nog_batch_refinement(
            cfg,
            {("NOG-FO", 0.01): rows},
        )
        self.assertEqual(manifest["base_candidate_count"], 1)
        self.assertEqual(manifest["variant_count"], 4)
        self.assertEqual(manifest["reused_variant_count"], 1)
        self.assertEqual(manifest["runnable_variant_count"], 3)
        self.assertEqual(manifest["runnable_task_count"], 9)

        validated = validate_nog_refinement_manifest(cfg, manifest)
        self.assertEqual(len(validated), 3)
        manifest["runnable_variants"][0]["parameters"]["eta"] = 999.0
        with self.assertRaisesRegex(ValueError, "changed outside batch grid"):
            validate_nog_refinement_manifest(cfg, manifest)

    def test_nog_refinement_work_audit_scales_with_smooth_batch(self):
        cfg = load_config(CONFIG)
        variant = {
            "candidate_id": "batch-four",
            "parameters": {
                "smooth_B": 4,
                "data_B_total": 64,
            },
        }
        expected = (960 + 2) * 4 * 64
        payloads = {
            "batch-four": [
                {
                    "formal_seed": 100,
                    "rows": [{"total_work": expected}],
                }
            ]
        }
        audit = nog_refinement_work_audit(cfg, [variant], payloads, 960)
        self.assertTrue(audit[0]["passed"])
        self.assertEqual(audit[0]["expected_final_total_work"], expected)


if __name__ == "__main__":
    unittest.main()
