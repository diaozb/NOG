from pathlib import Path

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.low_epsilon_analysis import _anomalies
from src.distributed.low_epsilon_freeze import _select_min_work_schedule
from src.distributed.low_epsilon_runner import (
    me_config,
    me_label,
    nog_config,
    nog_label,
)


CONFIG = Path("configs/distributed_cpu_fo_low_epsilon_v5.yaml")


def test_symmetric_low_epsilon_grid_and_seed_split_are_preregistered():
    cfg = load_config(str(CONFIG))
    ext = cfg["low_epsilon_extension"]
    epsilons = [float(value) for value in ext["epsilons"]]

    assert len(epsilons) == 25
    assert epsilons[0] == 0.01
    assert epsilons[-1] == 0.004
    assert epsilons == sorted(epsilons, reverse=True)
    assert all(
        abs((loose - strict) - 0.00025) < 1e-12
        for loose, strict in zip(epsilons, epsilons[1:])
    )
    assert set(ext["pilot_seeds"]).isdisjoint(cfg["run"]["formal_seeds"])
    assert set(ext["pilot_seeds"]).isdisjoint(
        ext["anomaly_confirmation"]["extra_formal_seeds"]
    )
    assert int(ext["max_parallel_tasks"]) * int(ext["reference_worker"]) == 32
    assert len(ext["algorithm_candidates"]["nog"]) == 6
    assert len(ext["algorithm_candidates"]["me_dol"]) == 6
    assert ext["batch_total_candidates"] == [32, 64, 96, 128, 192, 256]


def test_candidate_configs_share_depth_eval_grid_and_total_batch():
    cfg = load_config(str(CONFIG))
    nog = nog_config(cfg, 128, M=4, eta=0.5)
    me = me_config(cfg, 12, 150.0, 128)

    assert nog["methods"]["sfo"] == ["NOG-FO"]
    assert me["methods"]["sfo"] == ["ME-DOL-FO"]
    assert nog["train"] == me["train"] == {
        "rounds": 15360,
        "eval_every": 24,
        "strict_eval_grid": True,
    }
    assert nog["oracle"]["data_B_total"] == 128
    assert nog["nog"] == {"M": 4, "eta": 0.5}
    assert me["me_dol"] == {
        "epoch_length": 12,
        "theory_multiplier": 150.0,
        "smooth_B": 1,
        "data_B_per_worker": 16,
    }
    assert (
        nog_label(4, 0.5, 128, 15360)
        == "NOG-FO__M-4__eta-0p5__batch-total-128__rounds-15360"
    )
    assert (
        me_label(12, 150.0, 128, 15360)
        == "ME-DOL-FO__epoch-12__mult-150__batch-total-128__rounds-15360"
    )


def test_minimum_work_schedule_is_nondecreasing_and_can_switch():
    rows = [
        [
            {"batch_total": 32, "mean_total_work_hits": 100.0},
            {"batch_total": 64, "mean_total_work_hits": 120.0},
        ],
        [
            {"batch_total": 32, "mean_total_work_hits": 180.0},
            {"batch_total": 64, "mean_total_work_hits": 130.0},
        ],
    ]
    selected = _select_min_work_schedule(rows, switch_penalty=0.01)
    assert [row["batch_total"] for row in selected] == [32, 64]


def test_large_adjacent_depth_drop_is_flagged_without_parameter_reselection():
    rows = [
        {"epsilon": 0.01, "depth_ratio_mean": 2.0},
        {"epsilon": 0.00975, "depth_ratio_mean": 2.2},
        {"epsilon": 0.0095, "depth_ratio_mean": 1.6},
    ]

    anomalies = _anomalies(rows, 0.20)

    assert len(anomalies) == 1
    assert anomalies[0]["loose_epsilon"] == 0.00975
    assert anomalies[0]["strict_epsilon"] == 0.0095
    assert anomalies[0]["drop_fraction"] > 0.20
