from pathlib import Path

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.low_epsilon_v6_freeze import _best, _shortlist_rows
from src.distributed.low_epsilon_v6_analysis import _formal_label
from src.distributed.low_epsilon_v6_runner import candidate_map


CONFIG = Path("configs/distributed_cpu_fo_low_epsilon_v6.yaml")


def test_joint_grid_and_seed_isolation():
    cfg = load_config(CONFIG)
    ext = cfg["low_epsilon_extension"]
    candidates = candidate_map(cfg, int(ext["max_depth"]))

    # 16 NOG and 18 ME-DOL parameter choices, each at six common batches.
    assert len(candidates) == (16 + 18) * 6
    assert {row["method"] for row in candidates.values()} == {
        "NOG-FO",
        "ME-DOL-FO",
    }
    assert {row["batch_total"] for row in candidates.values()} == {
        8,
        16,
        32,
        64,
        128,
        256,
    }

    screen = set(ext["screen_seeds"])
    confirmation = set(ext["confirmation_pilot_seeds"])
    formal = set(cfg["run"]["formal_seeds"])
    assert not (screen & confirmation)
    assert not (screen & formal)
    assert not (confirmation & formal)


def test_joint_grid_uses_requested_depth_and_common_eval_grid():
    cfg = load_config(CONFIG)
    depth = int(cfg["low_epsilon_extension"]["max_depth"])
    candidates = candidate_map(cfg, depth)
    assert all(row["config"]["train"]["rounds"] == depth for row in candidates.values())
    assert all(row["config"]["train"]["eval_every"] == 24 for row in candidates.values())
    assert all(row["config"]["train"]["strict_eval_grid"] for row in candidates.values())


def test_shortlist_preserves_both_work_and_depth_candidates():
    rows = [
        {"label": "work", "hit_count": 3, "mean_total_work_hits": 10.0, "mean_depth_hits": 8.0},
        {"label": "depth", "hit_count": 3, "mean_total_work_hits": 20.0, "mean_depth_hits": 4.0},
        {"label": "other", "hit_count": 3, "mean_total_work_hits": 30.0, "mean_depth_hits": 30.0},
    ]
    labels = {row["label"] for row in _shortlist_rows(rows, 1, 0.0)}
    assert labels == {"work", "depth"}


def test_primary_regimes_optimize_independently():
    rows = [
        {"label": "work", "hit_count": 5, "mean_total_work_hits": 10.0, "mean_depth_hits": 8.0},
        {"label": "depth", "hit_count": 5, "mean_total_work_hits": 20.0, "mean_depth_hits": 4.0},
    ]
    assert _best(rows, "work_optimal", 5)["label"] == "work"
    assert _best(rows, "depth_optimal", 5)["label"] == "depth"


def test_formal_labels_replace_pilot_depth_without_changing_parameters():
    nog = {"method": "NOG-FO", "M": 2, "eta": 1.0, "batch_total": 16}
    me = {
        "method": "ME-DOL-FO",
        "epoch_length": 6,
        "theory_multiplier": 100.0,
        "batch_total": 32,
    }
    assert _formal_label(nog, 15360).endswith("batch-total-16__rounds-15360")
    assert _formal_label(me, 15360).endswith("batch-total-32__rounds-15360")
