import copy
import csv
import json
import unittest
from pathlib import Path

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.epsilon_scaling import (
    bootstrap_interval,
    epsilon_region,
    kaplan_meier_restricted_mean,
    paired_ratio_summary,
    summarize_censored,
    trend_statistics,
    validate_scaling_protocol,
)
from src.distributed.epsilon_scaling_protocol import prepare_protocol
from src.distributed.epsilon_scaling_dry_run import build_dry_run_config
from src.distributed.epsilon_scaling_pilot import pilot_runner_config
from src.distributed.epsilon_scaling_formal import build_formal_schedule, _task_config
from src.distributed.epsilon_scaling_robustness import (
    audit_robustness_preflight,
    prepare_robustness,
)
from src.distributed.epsilon_scaling_robustness_runner import (
    stage_entries,
    validate_manifest as validate_robustness_runner_manifest,
)
from src.distributed.cpu_fo_pilot import coarse_candidates, validate_pilot_config
from src.distributed.cpu_fo_tasks import file_sha256
from src.distributed.epsilon_scaling_pilot_analysis import (
    select_region,
    validate_refinement_manifest,
    validate_extension_manifest,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "distributed_cpu_fo_epsilon_scaling.yaml"


class EpsilonScalingTests(unittest.TestCase):
    def test_protocol_is_wide_deduplicated_and_seed_isolated(self):
        cfg = load_config(CONFIG)
        validate_scaling_protocol(cfg)
        self.assertEqual(len(cfg["epsilon_scaling"]["epsilons"]), 17)
        self.assertEqual(epsilon_region(0.1), "coarse")
        self.assertEqual(epsilon_region(0.01), "medium")
        self.assertEqual(epsilon_region(0.009), "fine")
        invalid = copy.deepcopy(cfg)
        invalid["epsilon_scaling"]["pilot_seeds"][0] = 0
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_scaling_protocol(invalid)

    def test_prepare_manifest_launches_nothing_and_counts_reuse(self):
        manifest = prepare_protocol(load_config(CONFIG))
        self.assertEqual(manifest["launches_started"], 0)
        self.assertEqual(manifest["primary_physical_task_count"], 120)
        self.assertEqual(manifest["primary_logical_method_epsilon_seed_rows"], 680)
        self.assertEqual(manifest["pilot_initial_physical_task_count"], 160)
        self.assertEqual(manifest["robustness_physical_task_count"], 240)
        self.assertEqual(len(manifest["primary_trajectory_groups"]), 6)

    def test_formal_schedule_is_frozen_unique_and_zero_launch(self):
        cfg = load_config(CONFIG)
        root = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
        schedule = build_formal_schedule(cfg, root)
        self.assertEqual(schedule["launches_started"], 0)
        self.assertEqual(schedule["task_count"], 120)
        self.assertEqual(schedule["max_concurrent_tasks"], 4)
        self.assertEqual(len({row["task_id"] for row in schedule["entries"]}), 120)
        fine = next(row for row in schedule["entries"] if row["group_id"] == "NOG-FO__fine")
        effective = _task_config(cfg, fine)
        self.assertEqual(effective["train"]["rounds"], 61440)
        self.assertEqual(effective["nog"]["M"], 24)
        self.assertEqual(effective["oracle"]["smooth_B"], 8)

    def test_robustness_manifest_reuses_formal_m8_and_stays_under_32_workers(self):
        cfg = load_config(CONFIG)
        root = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
        manifest = prepare_robustness(cfg, root)
        audit = audit_robustness_preflight(cfg, root, manifest)
        self.assertEqual(manifest["logical_task_count"], 240)
        self.assertEqual(manifest["runnable_task_count"], 180)
        self.assertEqual(manifest["reused_formal_task_count"], 60)
        self.assertEqual(manifest["launches_started"], 0)
        self.assertTrue(audit["launch_gate_passed"])
        self.assertTrue(
            all(
                int(worker) * int(limit) <= 32
                for worker, limit in manifest["max_concurrent_tasks_by_worker"].items()
            )
        )

    def test_robustness_runner_has_three_frozen_60_task_stages(self):
        cfg = load_config(CONFIG)
        root = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
        manifest = validate_robustness_runner_manifest(cfg, root)
        for worker, concurrency in ((1, 32), (2, 16), (4, 8)):
            entries = stage_entries(manifest, worker)
            self.assertEqual(len(entries), 60)
            self.assertTrue(all(row["worker_count"] == worker for row in entries))
            self.assertEqual(
                manifest["max_concurrent_tasks_by_worker"][str(worker)], concurrency
            )

    def test_formal_statistics_keep_all_seeds_and_nonhit_primary_values(self):
        cfg = load_config(CONFIG)
        analysis = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"] / "analysis"
        with open(analysis / "formal_statistics_manifest.json") as handle:
            manifest = json.load(handle)
        with open(analysis / "formal_per_seed.csv") as handle:
            per_seed = list(csv.DictReader(handle))
        with open(analysis / "formal_summary.csv") as handle:
            summaries = list(csv.DictReader(handle))
        with open(analysis / "formal_ratios.csv") as handle:
            ratios = list(csv.DictReader(handle))
        self.assertEqual(manifest["expanded_rows"], 680)
        self.assertEqual(len(per_seed), 680)
        self.assertEqual(len(summaries), 34)
        self.assertEqual(len(ratios), 17)
        self.assertTrue(manifest["nonhits_retained"])
        zero_hit = [row for row in summaries if row["hit_count"] == "0"]
        self.assertTrue(zero_hit)
        self.assertTrue(all(row["depth_capped_mean"] for row in zero_hit))
        self.assertTrue(all(row["depth_restricted_mean"] for row in zero_hit))
        self.assertTrue(all(row["depth_upper_bound_is_unbounded"] == "True" for row in zero_hit))

    def test_robustness_statistics_cover_grid_and_keep_zero_hits(self):
        cfg = load_config(CONFIG)
        analysis = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"] / "analysis"
        with open(analysis / "robustness_statistics_manifest.json") as handle:
            manifest = json.load(handle)
        with open(analysis / "robustness_per_seed.csv") as handle:
            per_seed = list(csv.DictReader(handle))
        with open(analysis / "robustness_summary.csv") as handle:
            summaries = list(csv.DictReader(handle))
        with open(analysis / "robustness_ratios.csv") as handle:
            ratios = list(csv.DictReader(handle))
        self.assertEqual(manifest["expanded_rows"], 240)
        self.assertEqual(len(per_seed), 240)
        self.assertEqual(len(summaries), 24)
        self.assertEqual(len(ratios), 12)
        self.assertTrue(manifest["nonhits_retained"])
        zero_hit = [row for row in summaries if row["hit_count"] == "0"]
        self.assertTrue(zero_hit)
        self.assertTrue(all(row["depth_capped_mean"] for row in zero_hit))
        nog_medium = [
            row
            for row in summaries
            if row["method"] == "NOG-FO" and row["epsilon"] == "0.01"
        ]
        self.assertEqual({row["hit_count"] for row in nog_medium}, {"10"})

    def test_wide_epsilon_report_and_figure_manifest_are_complete(self):
        cfg = load_config(CONFIG)
        root = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
        with open(root / "analysis" / "figure_report_manifest.json") as handle:
            manifest = json.load(handle)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(manifest["figures"]), 10)
        self.assertEqual(manifest["hypotheses"]["work_ratio_stability"], "not-supported")
        self.assertEqual(manifest["hypotheses"]["nonhit_reporting"], "supported")
        self.assertTrue(all((root / row["path"]).stat().st_size > 1000 for row in manifest["figures"]))
        report = (root / manifest["report"]).read_text(encoding="utf-8")
        self.assertIn("right-censored", report)
        self.assertIn("Partially supported", report)
        readme = (CONFIG.parents[1] / "README.md").read_text(encoding="utf-8")
        self.assertIn("2026 wide-epsilon 实验：当前主结果", readme)

    def test_compact_result_package_hashes_all_files(self):
        package = CONFIG.parents[1] / "results" / "epsilon_scaling_v2"
        with open(package / "package_manifest.json") as handle:
            manifest = json.load(handle)
        self.assertEqual(manifest["status"], "complete")
        self.assertFalse(manifest["raw_outputs_committed"])
        self.assertEqual(manifest["file_count"], len(manifest["files"]))
        self.assertLess(manifest["total_bytes"], 2 * 1024 * 1024)
        for row in manifest["files"]:
            path = package / row["path"]
            self.assertEqual(path.stat().st_size, row["bytes"])
            self.assertEqual(file_sha256(path), row["sha256"])

    def test_dry_run_config_is_small_and_keeps_all_epsilons(self):
        cfg = load_config(CONFIG)
        dry = build_dry_run_config(cfg)
        self.assertEqual(dry["train"]["rounds"], 24)
        self.assertEqual(dry["problem"]["d"], 8)
        self.assertEqual(dry["distributed"]["comparison_worker"], 2)
        self.assertEqual(dry["epsilon_scaling"]["epsilons"], cfg["epsilon_scaling"]["epsilons"])
        self.assertLessEqual(dry["cpu_process"]["max_total_worker_processes"], 4)

    def test_initial_pilot_uses_medium_batch_before_refinement(self):
        adapted = pilot_runner_config(load_config(CONFIG))
        validate_pilot_config(adapted)
        candidates = coarse_candidates(adapted)
        self.assertEqual(len(candidates), 32)
        self.assertEqual(sum(row["method"] == "NOG-FO" for row in candidates), 20)
        self.assertTrue(
            all(
                row["parameters"].get("smooth_B") == 2
                for row in candidates
                if row["method"] == "NOG-FO"
            )
        )

    def test_region_selection_uses_joint_coverage_not_one_epsilon(self):
        rows = [
            {
                "candidate_id": "stable",
                "coverage": 1.0,
                "worst_epsilon_hit_rate": 1.0,
                "full_coverage": True,
                "median_depth": 20,
                "median_total_work": 100,
                "median_training_time": 3,
                "median_final_stat_proxy": 0.01,
            },
            {
                "candidate_id": "unstable-fast",
                "coverage": 0.8,
                "worst_epsilon_hit_rate": 0.6,
                "full_coverage": False,
                "median_depth": 5,
                "median_total_work": 50,
                "median_training_time": 1,
                "median_final_stat_proxy": 0.005,
            },
        ]
        selected = select_region(rows, top_n_censored=1)
        self.assertEqual(selected["selected_candidate_id"], "stable")
        self.assertEqual(selected["status"], "full-coverage")

    def test_refinement_manifest_rejects_tampering(self):
        from src.distributed.cpu_fo_tasks import object_sha256

        cfg = load_config(CONFIG)
        parameters = {"M": 4, "eta": 0.1, "smooth_B": 1, "data_B_total": 64}
        variant = {
            "candidate_id": "NOG-FO__M-4__data_B_total-64__eta-0p1__smooth_B-1",
            "base_candidate_id": "base",
            "method": "NOG-FO",
            "parameters": parameters,
            "reuse_existing": False,
        }
        manifest = {
            "status": "prepared",
            "launches_started": 0,
            "runnable_variants": [variant],
            "runnable_task_count": 5,
        }
        manifest["manifest_sha256"] = object_sha256(manifest)
        self.assertEqual(len(validate_refinement_manifest(cfg, manifest)), 1)
        manifest["runnable_variants"][0]["parameters"]["smooth_B"] = 999
        with self.assertRaisesRegex(ValueError, "SHA256"):
            validate_refinement_manifest(cfg, manifest)

    def test_extension_manifest_requires_adjacent_budget_and_hash(self):
        from src.distributed.cpu_fo_tasks import object_sha256

        cfg = load_config(CONFIG)
        manifest = {
            "status": "prepared",
            "launches_started": 0,
            "source_rounds": 960,
            "target_rounds": 3840,
            "candidates": [],
            "task_count": 0,
        }
        manifest["manifest_sha256"] = object_sha256(manifest)
        self.assertEqual(validate_extension_manifest(cfg, manifest), [])
        manifest["target_rounds"] = 61440
        manifest["manifest_sha256"] = object_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        with self.assertRaisesRegex(ValueError, "adjacent"):
            validate_extension_manifest(cfg, manifest)

    def test_censoring_summary_never_drops_nonhits(self):
        summary = summarize_censored([10.0, None, 30.0], [40.0, 40.0, 40.0])
        self.assertEqual(summary["hit_count"], 2)
        self.assertAlmostEqual(summary["hit_rate"], 2 / 3)
        self.assertAlmostEqual(summary["conditional_mean"], 20.0)
        self.assertAlmostEqual(summary["capped_mean"], 80 / 3)
        self.assertGreaterEqual(summary["restricted_mean"], summary["conditional_mean"])

    def test_kaplan_meier_all_hits_matches_mean(self):
        self.assertAlmostEqual(
            kaplan_meier_restricted_mean([10.0, 20.0, 30.0], [True, True, True], 30.0),
            20.0,
        )

    def test_paired_ratio_marks_partial_hits_conditional(self):
        summary = paired_ratio_summary(
            [20.0, None, 60.0],
            [10.0, 20.0, None],
            [80.0, 80.0, 80.0],
            [40.0, 40.0, 40.0],
        )
        self.assertEqual(summary["paired_hit_count"], 1)
        self.assertEqual(summary["paired_hit_ratio_median"], 2.0)
        self.assertTrue(summary["ratios_are_conditional"])
        self.assertIsNotNone(summary["ratio_of_capped_means"])

    def test_bootstrap_is_deterministic_and_trend_detects_growth(self):
        first = bootstrap_interval([1.0, 2.0, 3.0], 200, 7)
        self.assertEqual(first, bootstrap_interval([1.0, 2.0, 3.0], 200, 7))
        trend = trend_statistics([0.1, 0.01, 0.001], [1.0, 2.0, 4.0])
        self.assertGreater(trend["log_log_slope"], 0.0)
        self.assertAlmostEqual(trend["spearman_rho"], 1.0)


if __name__ == "__main__":
    unittest.main()
