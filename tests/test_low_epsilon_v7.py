from pathlib import Path

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.low_epsilon_v7_analysis import _segment
from src.distributed.low_epsilon_v7_freeze import _candidate_summary, _select
from src.distributed.low_epsilon_v7_runner import (
    formal_descriptors,
    me_config,
    nog_config,
    pilot_descriptors,
)


CONFIG = Path("configs/distributed_cpu_fo_low_epsilon_v7.yaml")


def test_theory_schedule_covers_every_epsilon_once():
    cfg = load_config(CONFIG)
    ext = cfg["theory_scaling"]
    for epsilon in ext["epsilons"]:
        assert _segment(ext["nog_segments"], float(epsilon))["id"].startswith("n")
        assert _segment(ext["me_segments"], float(epsilon))["id"].startswith("m")


def test_theory_schedule_scales_in_the_preregistered_directions():
    cfg = load_config(CONFIG)
    ext = cfg["theory_scaling"]
    nog = ext["nog_segments"]
    me = ext["me_segments"]
    assert [row["M"] for row in nog] == sorted(row["M"] for row in nog)
    assert [row["eta"] for row in nog] == sorted(
        (row["eta"] for row in nog), reverse=True
    )
    assert [row["batch_total"] for row in nog] == sorted(
        row["batch_total"] for row in nog
    )
    assert [row["epoch_length"] for row in me] == sorted(
        row["epoch_length"] for row in me
    )
    assert {row["batch_total"] for row in me} == {8}


def test_pilot_grid_and_seed_isolation():
    cfg = load_config(CONFIG)
    ext = cfg["theory_scaling"]
    candidates = pilot_descriptors(cfg)
    assert len(candidates) == 3 * 6 + 5 * 6
    assert {row["method"] for row in candidates.values()} == {
        "NOG-FO",
        "ME-DOL-FO",
    }
    assert not (set(ext["pilot_seeds"]) & set(cfg["run"]["formal_seeds"]))


def test_candidate_configs_use_natural_theory_periods_and_exact_work_batches():
    cfg = load_config(CONFIG)
    ext = cfg["theory_scaling"]
    depth = int(ext["pilot_max_depth"])
    nseg, mseg = ext["nog_segments"][-1], ext["me_segments"][-1]
    ncfg = nog_config(cfg, nseg, 1.0, depth)
    mcfg = me_config(cfg, mseg, 100.0, depth)
    assert ncfg["train"]["eval_every"] == nseg["M"]
    assert ncfg["oracle"]["data_B_total"] == nseg["batch_total"]
    assert mcfg["train"]["eval_every"] == mseg["epoch_length"]
    assert mcfg["me_dol"]["data_B_per_worker"] == 1
    assert not ncfg["train"]["strict_eval_grid"]
    assert not mcfg["train"]["strict_eval_grid"]


def test_freeze_selects_only_full_anchor_coverage_then_minimum_work():
    measurements = [
        {
            "method": "NOG-FO",
            "eta_scale": scale,
            "hit_count": hits,
            "mean_total_work_hits": work,
            "mean_depth_hits": work / 8,
        }
        for scale, hits, work in [(0.5, 5, 20.0), (0.5, 5, 20.0), (1.0, 5, 10.0), (1.0, 5, 10.0)]
    ]
    summary = _candidate_summary(measurements, "NOG-FO", "eta_scale", 5)
    assert _select(summary, 2)["constant_value"] == 1.0


def test_formal_grid_has_six_segments_per_method():
    cfg = load_config(CONFIG)
    freeze = {
        "selected_global_constants": {
            "NOG-FO": {"eta_scale": 1.0},
            "ME-DOL-FO": {"theory_multiplier": 100.0},
        }
    }
    rows = formal_descriptors(cfg, freeze)
    assert len(rows) == 12
    assert sum(row["method"] == "NOG-FO" for row in rows) == 6
    assert sum(row["method"] == "ME-DOL-FO" for row in rows) == 6
