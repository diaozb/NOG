import copy

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.theory_validation_v4_extension_runner import me_config, nog_config, stage_configs


CONFIG = "configs/distributed_cpu_fo_theory_validation_v4_extended_budget.yaml"


def test_extension_changes_only_numerical_budget_and_keeps_v4_parameters():
    base = load_config(CONFIG)
    original = copy.deepcopy(base)
    nog = nog_config(base, 3840)
    me = me_config(base, 15360)
    assert base == original
    assert nog["train"] == {"rounds": 3840, "eval_every": 2}
    assert nog["oracle"]["smooth_B"] == 1
    assert nog["oracle"]["data_B_total"] == 16
    assert nog["nog"] == {"M": 2, "eta": 1.0}
    assert nog["methods"]["sfo"] == ["NOG-FO"]
    assert me["train"] == {"rounds": 15360, "eval_every": 6}
    assert me["oracle"]["smooth_B"] == 2
    assert me["oracle"]["data_B_total"] == 64
    assert me["me_dol"] == {"epoch_length": 6, "theory_multiplier": 100.0}
    assert me["methods"]["sfo"] == ["ME-DOL-FO"]


def test_stage_budgets_expand_by_four():
    cfg = load_config(CONFIG)
    first = stage_configs(cfg, "stage1")
    second = stage_configs(cfg, "stage2")
    for method in ["NOG-FO", "ME-DOL-FO"]:
        assert second[method]["train"]["rounds"] == 4 * first[method]["train"]["rounds"]


def test_dense_low_epsilon_grid_is_descending_and_unique():
    cfg = load_config(CONFIG)
    eps = cfg["extended_budget"]["epsilons"]
    assert eps == sorted(set(eps), reverse=True)
    assert max(eps) < 0.01
    assert min(eps) == 0.002
