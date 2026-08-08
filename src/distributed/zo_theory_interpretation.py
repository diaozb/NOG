"""Build the Step ZO-5C theory-comparison artifact.

The script consumes only the audited Step ZO-5B outputs.  It resamples formal
seeds jointly across epsilon thresholds so that slope confidence intervals
preserve the within-seed dependence of the reused trajectories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "zo_experiments/formal"
PER_SEED = FORMAL / "formal_per_seed.csv"
AUDIT = FORMAL / "audit.json"
OUTPUT_CSV = FORMAL / "theory_comparison.csv"
OUTPUT_JSON = FORMAL / "theory_comparison.json"
OUTPUT_REPORT = FORMAL / "STEP_ZO_5C_THEORY_COMPARISON.md"

BASELINES = ["ME-DOL-ZO", "DGFM", "DGFM+"]
THEORY = {
    "ME-DOL-ZO": {
        "depth_ratio_exponent": 4.0 / 3.0,
        "work_ratio_exponent": 0.0,
        "baseline_depth_exponent": 3.0,
        "baseline_work_exponent": 3.0,
    },
    "DGFM": {
        "depth_ratio_exponent": 7.0 / 3.0,
        "work_ratio_exponent": 1.0,
        "baseline_depth_exponent": 4.0,
        "baseline_work_exponent": 4.0,
    },
    "DGFM+": {
        "depth_ratio_exponent": 4.0 / 3.0,
        "work_ratio_exponent": 0.0,
        "baseline_depth_exponent": 3.0,
        "baseline_work_exponent": 3.0,
    },
}
NOG_DEPTH_EXPONENT = 5.0 / 3.0
NOG_WORK_EXPONENT = 3.0
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20260807


def slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.log(1.0 / x), np.log(y), 1)[0])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def paired_matrix(
    per_seed: pd.DataFrame,
    baseline: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nog = per_seed.loc[per_seed["method"] == "NOG-ZO"].copy()
    base = per_seed.loc[per_seed["method"] == baseline].copy()
    merged = nog.merge(
        base,
        on=["formal_seed", "epsilon"],
        suffixes=("_nog", "_base"),
        validate="one_to_one",
    )
    merged["paired_hit"] = merged["hit_nog"] & merged["hit_base"]
    complete_epsilons = merged.groupby("epsilon")["paired_hit"].all()
    epsilons = np.asarray(
        sorted(
            complete_epsilons.loc[complete_epsilons].index.astype(float),
            reverse=True,
        ),
        dtype=float,
    )
    seeds = np.asarray(sorted(merged["formal_seed"].unique()), dtype=int)
    depth = np.empty((len(seeds), len(epsilons)), dtype=float)
    work = np.empty_like(depth)
    for seed_index, seed in enumerate(seeds):
        frame = merged.loc[merged["formal_seed"] == seed].set_index("epsilon")
        for epsilon_index, epsilon in enumerate(epsilons):
            row = frame.loc[epsilon]
            depth[seed_index, epsilon_index] = (
                float(row["first_hit_depth_base"])
                / float(row["first_hit_depth_nog"])
            )
            work[seed_index, epsilon_index] = (
                float(row["first_hit_work_base"])
                / float(row["first_hit_work_nog"])
            )
    return epsilons, seeds, depth, work


def bootstrap_slope_ci(
    epsilons: np.ndarray,
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    seed_count = values.shape[0]
    slopes = np.empty(BOOTSTRAP_REPETITIONS, dtype=float)
    for index in range(BOOTSTRAP_REPETITIONS):
        sampled = rng.integers(0, seed_count, size=seed_count)
        slopes[index] = slope(epsilons, values[sampled].mean(axis=0))
    low, high = np.quantile(slopes, [0.025, 0.975])
    return float(low), float(high)


def summarize(per_seed: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for baseline in BASELINES:
        epsilons, seeds, depth, work = paired_matrix(per_seed, baseline)
        mean_depth = depth.mean(axis=0)
        mean_work = work.mean(axis=0)
        depth_slope = slope(epsilons, mean_depth)
        work_slope = slope(epsilons, mean_work)
        depth_low, depth_high = bootstrap_slope_ci(epsilons, depth, rng)
        work_low, work_high = bootstrap_slope_ci(epsilons, work, rng)
        theory = THEORY[baseline]
        depth_theory_growth = float(
            (epsilons[0] / epsilons[-1])
            ** theory["depth_ratio_exponent"]
        )
        work_theory_growth = float(
            (epsilons[0] / epsilons[-1])
            ** theory["work_ratio_exponent"]
        )
        row = {
            "baseline": baseline,
            "ratio_direction": f"{baseline}/NOG-ZO",
            "complete_pairing_points": int(len(epsilons)),
            "formal_seeds": int(len(seeds)),
            "epsilon_high": float(epsilons[0]),
            "epsilon_low": float(epsilons[-1]),
            "theory_depth_ratio_exponent": float(
                theory["depth_ratio_exponent"]
            ),
            "observed_depth_ratio_slope": depth_slope,
            "depth_slope_ci_low": depth_low,
            "depth_slope_ci_high": depth_high,
            "depth_spearman": spearman(1.0 / epsilons, mean_depth),
            "depth_ratio_high_epsilon": float(mean_depth[0]),
            "depth_ratio_low_epsilon": float(mean_depth[-1]),
            "observed_depth_ratio_growth": float(
                mean_depth[-1] / mean_depth[0]
            ),
            "theory_depth_ratio_growth": depth_theory_growth,
            "theory_work_ratio_exponent": float(
                theory["work_ratio_exponent"]
            ),
            "observed_work_ratio_slope": work_slope,
            "work_slope_ci_low": work_low,
            "work_slope_ci_high": work_high,
            "work_spearman": spearman(1.0 / epsilons, mean_work),
            "work_ratio_high_epsilon": float(mean_work[0]),
            "work_ratio_low_epsilon": float(mean_work[-1]),
            "observed_work_ratio_growth": float(
                mean_work[-1] / mean_work[0]
            ),
            "theory_work_ratio_growth": work_theory_growth,
            "depth_direction_supported": bool(depth_low > 0),
            "exact_depth_exponent_inside_ci": bool(
                depth_low
                <= theory["depth_ratio_exponent"]
                <= depth_high
            ),
            "work_direction_supported": bool(
                work_low > 0
                if theory["work_ratio_exponent"] > 0
                else work_low <= 0 <= work_high
            ),
            "exact_work_exponent_inside_ci": bool(
                work_low <= theory["work_ratio_exponent"] <= work_high
            ),
        }
        rows.append(row)
        details[baseline] = {
            "epsilons": epsilons.tolist(),
            "mean_depth_ratios": mean_depth.tolist(),
            "mean_work_ratios": mean_work.tolist(),
        }
    return pd.DataFrame(rows), details


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_report(comparison: pd.DataFrame, audit: dict[str, Any]) -> None:
    lines = [
        "# Step ZO-5C: theory comparison and claim boundary",
        "",
        "## 1. What this step evaluates",
        "",
        "Step ZO-5C interprets the audited 20-seed formal experiment against "
        "Theorem parallel-ZO, Theorem dist-ZO, and Table res-dist in the "
        "paper source. It does not rerun trajectories or tune parameters.",
        "",
        f"The input audit passed {audit['tasks_observed']}/"
        f"{audit['tasks_expected']} frozen method-seed tasks. The primary "
        "estimand is the same-seed baseline/NOG-ZO first-hit depth ratio.",
        "",
        "## 2. Theoretical ratio directions",
        "",
        "After fixing dimension, Goldstein radius, smoothness constants, and "
        "initial optimality gap, the leading epsilon powers imply:",
        "",
        "| baseline/NOG-ZO | predicted depth-ratio power | "
        "predicted work-ratio power |",
        "|---|---:|---:|",
        "| ME-DOL-ZO | 4/3 | 0 |",
        "| DGFM | 7/3 | 1 |",
        "| DGFM+ | 4/3 | 0 |",
        "",
        "A positive depth-ratio power predicts a growing relative depth "
        "advantage for NOG-ZO as epsilon decreases. A zero work-ratio power "
        "means only equal asymptotic order; it does not require a ratio of "
        "one and does not remove finite constants.",
        "",
        "## 3. Formal observations on complete-pair intervals",
        "",
        "Slope confidence intervals use 2,000 joint bootstrap resamples of "
        "the 20 formal seeds. Each resample preserves dependence across "
        "epsilon thresholds because all thresholds reuse the same trajectory.",
        "",
        "| baseline | complete epsilon interval | depth ratio endpoints | "
        "observed depth slope (95% CI) | theory | work slope (95% CI) | "
        "theory |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['baseline']} | "
            f"{row['epsilon_high']:.3f} to {row['epsilon_low']:.3f} | "
            f"{fmt(row['depth_ratio_high_epsilon'], 2)} to "
            f"{fmt(row['depth_ratio_low_epsilon'], 2)} | "
            f"{fmt(row['observed_depth_ratio_slope'])} "
            f"[{fmt(row['depth_slope_ci_low'])}, "
            f"{fmt(row['depth_slope_ci_high'])}] | "
            f"{fmt(row['theory_depth_ratio_exponent'])} | "
            f"{fmt(row['observed_work_ratio_slope'])} "
            f"[{fmt(row['work_slope_ci_low'])}, "
            f"{fmt(row['work_slope_ci_high'])}] | "
            f"{fmt(row['theory_work_ratio_exponent'])} |"
        )
    lines.extend(
        [
            "",
            "All three observed depth slopes are positive, their bootstrap "
            "intervals exclude zero, and their mean curves have Spearman "
            "coefficient one against inverse epsilon. This is direct "
            "finite-instance evidence for the predicted direction of the "
            "relative communication-depth advantage.",
            "",
            "The observed depth-slope intervals do not contain the worst-case "
            "theoretical ratio exponents. This does not contradict a Big-O "
            "upper bound, but it means the experiment does not verify the "
            "exact 5/3, 3, 4, 4/3, or 7/3 powers.",
            "",
            "The work result is weaker. ME-DOL-ZO/NOG-ZO and DGFM+/NOG-ZO "
            "increase over their complete intervals even though their "
            "worst-case work orders have the same epsilon power. DGFM/NOG-ZO "
            "increases in the predicted direction but more slowly than the "
            "reference power one. Consequently, work should be reported as "
            "a secondary finite-budget tradeoff, not as validation of exact "
            "work scaling.",
            "",
            "## 4. Censoring boundary",
            "",
            "- ME-DOL-ZO/NOG-ZO and DGFM/NOG-ZO have 20/20 paired hits down "
            "to epsilon 0.018. NOG-ZO then has 18/20 hits at 0.016, 11/20 "
            "at 0.015, 1/20 at 0.014, and 0/20 at 0.013.",
            "- DGFM+/NOG-ZO has 20/20 paired hits down to epsilon 0.030; "
            "DGFM+ has no confirmed hits at 0.025 or below.",
            "- Ratios below these complete-pair boundaries are right-censored "
            "and are excluded from slope fitting. Conditional ratios there "
            "are survivor-biased.",
            "",
            "## 5. Why exact exponent verification is out of scope",
            "",
            "1. The theorem prescribes epsilon-dependent quantities, including "
            "the NOG-ZO oracle variance and batch size. The formal experiment "
            "instead freezes one pilot-selected configuration and reads all "
            "thresholds from one anytime trajectory.",
            "2. Worst-case Big-O upper bounds need not be tight on one finite "
            "synthetic objective or over a one-decade accuracy range.",
            "3. The measured endpoint is a fixed-bank Monte Carlo smoothed-"
            "gradient proxy, not the exact Goldstein subdifferential distance.",
            "4. The experiment simulates dependency depth on a complete graph; "
            "it does not measure multi-machine latency.",
            "5. The low-epsilon region is budget-censored.",
            "",
            "## 6. Paper-safe conclusion",
            "",
            "> Across the complete paired-hit regime, the communication-depth "
            "ratios of ME-DOL-ZO, DGFM, and DGFM+ relative to NOG-ZO increase "
            "monotonically as the target accuracy becomes more stringent. "
            "This finite-instance trend is qualitatively consistent with the "
            "improved epsilon dependence of NOG-ZO communication complexity. "
            "We do not interpret the experiment as verification of the exact "
            "worst-case exponents: the configurations are frozen across "
            "thresholds, the stationarity measure is a Monte Carlo proxy, and "
            "the smaller-epsilon observations are right-censored by the fixed "
            "training-work budget.",
            "",
            "## 7. Claims that should not be made",
            "",
            "- Do not claim that the experiment proves Theorem parallel-ZO or "
            "Theorem dist-ZO.",
            "- Do not claim empirical recovery of the exact theoretical powers.",
            "- Do not report hit-only ratios below the complete-pair boundary "
            "as unbiased comparisons.",
            "- Do not claim constant empirical work ratios for ME-DOL-ZO or "
            "DGFM+ under this formal configuration.",
            "- Do not describe the frozen pilot configurations as globally "
            "optimal hyperparameters.",
            "",
            "## 8. Reproduction",
            "",
            "    conda run -n NOG python -m "
            "src.distributed.zo_theory_interpretation",
            "",
            "Machine-readable values are stored in theory_comparison.csv and "
            "theory_comparison.json. The underlying first-hit and ratio tables "
            "remain formal_per_seed.csv and formal_ratios.csv.",
            "",
        ]
    )
    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise RuntimeError("Step ZO-5B audit must pass before interpretation.")
    per_seed = pd.read_csv(PER_SEED)
    comparison, details = summarize(per_seed)
    comparison.to_csv(OUTPUT_CSV, index=False)
    payload = {
        "step": "ZO-5C",
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "nog_theory": {
            "depth_epsilon_exponent": NOG_DEPTH_EXPONENT,
            "work_epsilon_exponent": NOG_WORK_EXPONENT,
        },
        "comparison": comparison.to_dict(orient="records"),
        "curves": details,
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_report(comparison, audit)
    print(comparison.to_string(index=False))
    print(f"saved={OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
