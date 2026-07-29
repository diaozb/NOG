import copy

import pytest

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.theory_validation_runner import me_formal_config, nog_batch_config


def test_nog_batch_config_isolated_and_frozen() -> None:
    base = load_config("configs/distributed_cpu_fo_theory_validation_v4.yaml")
    original = copy.deepcopy(base)
    cfg = nog_batch_config(base, 24)
    assert base == original
    assert cfg["oracle"]["data_B_total"] == 24
    assert cfg["oracle"]["smooth_B"] == 1
    assert cfg["nog"] == {"M": 2, "eta": 1.0}
    assert cfg["train"] == {"rounds": 960, "eval_every": 2}
    assert cfg["methods"]["sfo"] == ["NOG-FO"]


@pytest.mark.parametrize("batch", [0, 7, 9, 63])
def test_nog_batch_config_requires_multiple_of_workers(batch: int) -> None:
    base = load_config("configs/distributed_cpu_fo_theory_validation_v4.yaml")
    with pytest.raises(ValueError, match="multiple of 8"):
        nog_batch_config(base, batch)


def test_me_formal_config_isolated_and_extended() -> None:
    base = load_config("configs/distributed_cpu_fo_theory_validation_v4.yaml")
    original = copy.deepcopy(base)
    cfg = me_formal_config(base)
    assert base == original
    assert cfg["train"] == {"rounds": 3840, "eval_every": 6}
    assert cfg["me_dol"] == {"epoch_length": 6, "theory_multiplier": 100.0}
    assert cfg["methods"]["sfo"] == ["ME-DOL-FO"]
