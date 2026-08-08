"""Step ZO-6C: compare reserved anomaly seeds with formal trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.distributed.zo_anomaly_audit import task_diagnostics


ROOT = Path(__file__).resolve().parents[2]
FORMAL_RAW = (
    ROOT
    / "outputs/distributed_zo/zo_theory_validation/formal/fixed_work_983040"
    / "results.csv"
)
DIAGNOSTIC = (
    ROOT
    / "outputs/distributed_zo/zo_theory_validation/diagnostic"
    / "anomaly_seeds_fixed_work_983040"
)
DIAGNOSTIC_RAW = DIAGNOSTIC / "results.csv"
PROGRESS = DIAGNOSTIC / "progress.json"
MANIFEST = DIAGNOSTIC / "diagnostic_manifest.json"
FREEZE = DIAGNOSTIC / "frozen_parameters.json"
OUTPUT = ROOT / "zo_experiments/formal"
FIGURES = OUTPUT / "figures"
CSV_OUTPUT = OUTPUT / "anomaly_replication_comparison.csv"
JSON_OUTPUT = OUTPUT / "anomaly_replication_comparison.json"
REPORT_OUTPUT = OUTPUT / "STEP_ZO_6C_REPLICATION_DECISION.md"

METHODS = ["NOG-ZO", "ME-DOL-ZO", "DGFM", "DGFM+"]
METRICS = [
    "confirmed_floor",
    "confirmed_floor_work_fraction",
    "final_over_confirmed_floor",
]
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20260808


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: str) -> str:
    return json.dumps(json.loads(value), sort_keys=True, separators=(",", ":"))


def audit_diagnostic(results: pd.DataFrame) -> dict[str, Any]:
    progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    errors: list[str] = []
    if progress.get("status") != "complete":
        errors.append("diagnostic progress is not complete")
    if manifest.get("merge_with_formal_results") is not False:
        errors.append("diagnostic manifest does not prohibit formal merging")
    seeds = {int(value) for value in manifest["anomaly_seeds"]}
    if seeds != {200, 201, 202, 203, 204}:
        errors.append(f"unexpected anomaly seeds: {sorted(seeds)}")
    prior = {
        int(value)
        for key in ["pilot_seeds", "formal_seeds"]
        for value in manifest[key]
    }
    if seeds.intersection(prior):
        errors.append("anomaly seeds overlap pilot/formal seeds")
    selected = {
        entry["method"]: json.dumps(
            entry["parameters"], sort_keys=True, separators=(",", ":")
        )
        for entry in freeze["selected_candidates"]
    }
    expected = {(method, seed) for method in selected for seed in seeds}
    observed = {
        (str(method), int(seed))
        for method, seed in results[["method", "formal_seed"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    if observed != expected:
        errors.append(
            f"task mismatch missing={sorted(expected-observed)} "
            f"extra={sorted(observed-expected)}"
        )
    target_work = int(manifest["target_total_work"])
    for (method, seed), frame in results.groupby(
        ["method", "formal_seed"], sort=True
    ):
        ordered = frame.sort_values("depth")
        parameters = {
            canonical(value)
            for value in ordered["candidate_parameters"].astype(str)
        }
        if parameters != {selected[str(method)]}:
            errors.append(f"{method}/seed-{seed}: parameter mismatch")
        if not np.all(np.diff(ordered["depth"].to_numpy(dtype=float)) > 0):
            errors.append(f"{method}/seed-{seed}: non-increasing depth")
        if not np.all(
            np.diff(ordered["total_work"].to_numpy(dtype=float)) > 0
        ):
            errors.append(f"{method}/seed-{seed}: non-increasing work")
        if int(ordered["total_work"].iloc[-1]) != target_work:
            errors.append(f"{method}/seed-{seed}: final work mismatch")
        proxy = ordered["stat_proxy"].to_numpy(dtype=float)
        if not np.all(np.isfinite(proxy)) or np.any(proxy < 0):
            errors.append(f"{method}/seed-{seed}: invalid proxy")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "tasks_expected": len(expected),
        "tasks_observed": len(observed),
        "rows": int(len(results)),
        "anomaly_seeds": sorted(seeds),
        "target_total_work": target_work,
        "results_sha256": sha256(DIAGNOSTIC_RAW),
        "formal_results_sha256": sha256(FORMAL_RAW),
        "freeze_sha256": sha256(FREEZE),
    }


def difference_ci(
    formal: np.ndarray,
    anomaly: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.empty(BOOTSTRAP_REPETITIONS, dtype=float)
    for index in range(BOOTSTRAP_REPETITIONS):
        f = rng.choice(formal, size=len(formal), replace=True)
        a = rng.choice(anomaly, size=len(anomaly), replace=True)
        values[index] = a.mean() - f.mean()
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def comparison_table(
    formal_tasks: pd.DataFrame,
    anomaly_tasks: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    records: list[dict[str, Any]] = []
    for method in METHODS:
        formal = formal_tasks.loc[formal_tasks["method"] == method]
        anomaly = anomaly_tasks.loc[anomaly_tasks["method"] == method]
        record: dict[str, Any] = {
            "method": method,
            "formal_seeds": int(len(formal)),
            "anomaly_seeds": int(len(anomaly)),
            "formal_early_rebound_rate": float(formal["early_rebound"].mean()),
            "anomaly_early_rebound_rate": float(
                anomaly["early_rebound"].mean()
            ),
            "formal_late_floor_rate": float(formal["late_floor"].mean()),
            "anomaly_late_floor_rate": float(anomaly["late_floor"].mean()),
        }
        for metric in METRICS:
            fvalues = formal[metric].to_numpy(dtype=float)
            avalues = anomaly[metric].to_numpy(dtype=float)
            low, high = difference_ci(fvalues, avalues, rng)
            record[f"formal_mean_{metric}"] = float(fvalues.mean())
            record[f"anomaly_mean_{metric}"] = float(avalues.mean())
            record[f"difference_{metric}"] = float(
                avalues.mean() - fvalues.mean()
            )
            record[f"difference_{metric}_ci_low"] = low
            record[f"difference_{metric}_ci_high"] = high
        if method == "NOG-ZO":
            replicated = bool(
                formal["early_rebound"].all()
                and anomaly["early_rebound"].all()
            )
            decision = "do_not_extend_same_frozen_configuration"
        else:
            replicated = bool(
                not formal["early_rebound"].any()
                and not anomaly["early_rebound"].any()
            )
            decision = "no_extension_needed_for_current_paper_claim"
        record["trajectory_pattern_replicated"] = replicated
        record["budget_extension_decision"] = decision
        records.append(record)
    return pd.DataFrame(records)


def make_figure(
    formal_tasks: pd.DataFrame,
    anomaly_tasks: pd.DataFrame,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = {
        "confirmed_floor": "confirmed proxy floor",
        "confirmed_floor_work_fraction": "work fraction at floor",
        "final_over_confirmed_floor": "final proxy / floor",
    }
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for axis, metric in zip(axes, METRICS):
        positions: list[float] = []
        values: list[np.ndarray] = []
        colors: list[str] = []
        tick_positions: list[float] = []
        for method_index, method in enumerate(METHODS):
            center = method_index * 3.0
            tick_positions.append(center + 0.5)
            fvalues = formal_tasks.loc[
                formal_tasks["method"] == method, metric
            ].to_numpy(dtype=float)
            avalues = anomaly_tasks.loc[
                anomaly_tasks["method"] == method, metric
            ].to_numpy(dtype=float)
            positions.extend([center, center + 1.0])
            values.extend([fvalues, avalues])
            colors.extend(["#9ecae1", "#fdae6b"])
        boxes = axis.boxplot(
            values,
            positions=positions,
            widths=0.65,
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        for position, group in zip(positions, values):
            jitter = rng.normal(0.0, 0.055, size=len(group))
            axis.scatter(
                np.full(len(group), position) + jitter,
                group,
                color="black",
                alpha=0.55,
                s=12,
                zorder=3,
            )
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(METHODS, rotation=20)
        axis.set_ylabel(labels[metric])
        axis.grid(True, axis="y", alpha=0.25)
        axis.plot([], [], color="#9ecae1", linewidth=8, label="formal 20")
        axis.plot([], [], color="#fdae6b", linewidth=8, label="anomaly 5")
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Frozen-configuration replication: formal versus reserved seeds"
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "anomaly_replication_comparison.png", dpi=220)
    figure.savefig(FIGURES / "anomaly_replication_comparison.pdf")
    plt.close(figure)


def write_report(
    comparison: pd.DataFrame,
    audit: dict[str, Any],
) -> None:
    lines = [
        "# Step ZO-6C: replication result and budget decision",
        "",
        "## 1. Integrity",
        "",
        f"The diagnostic audit passed {audit['tasks_observed']}/"
        f"{audit['tasks_expected']} tasks on reserved seeds "
        f"{audit['anomaly_seeds']}. Each task reused the frozen formal "
        f"candidate and ended at work {audit['target_total_work']:,}. "
        "Diagnostic results remain separate from the 20 formal seeds.",
        "",
        "## 2. Formal-versus-anomaly comparison",
        "",
        "| method | confirmed floor: formal / anomaly | floor position: "
        "formal / anomaly | final/floor: formal / anomaly | early rebound: "
        "formal / anomaly | pattern replicated |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['method']} | "
            f"{row['formal_mean_confirmed_floor']:.5f} / "
            f"{row['anomaly_mean_confirmed_floor']:.5f} | "
            f"{100 * row['formal_mean_confirmed_floor_work_fraction']:.1f}% / "
            f"{100 * row['anomaly_mean_confirmed_floor_work_fraction']:.1f}% | "
            f"{row['formal_mean_final_over_confirmed_floor']:.2f} / "
            f"{row['anomaly_mean_final_over_confirmed_floor']:.2f} | "
            f"{100 * row['formal_early_rebound_rate']:.0f}% / "
            f"{100 * row['anomaly_early_rebound_rate']:.0f}% | "
            f"{'yes' if row['trajectory_pattern_replicated'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "![Formal and anomaly comparison](figures/anomaly_replication_comparison.png)",
            "",
            "## 3. Replication conclusion",
            "",
            "The NOG-ZO anomaly replicates exactly under the preregistered "
            "rule: all 20 formal seeds and all five reserved anomaly seeds "
            "reach their confirmed floor before half of the work budget and "
            "finish at more than 1.5 times that floor. The anomaly-seed mean "
            "floor position and final/floor ratio are close to the formal "
            "values. This is configuration-level late-run instability rather "
            "than an idiosyncrasy of the original formal seeds.",
            "",
            "ME-DOL-ZO, DGFM, and DGFM+ replicate the contrasting pattern: no "
            "formal or anomaly seed satisfies the early-rebound definition. "
            "Their floors occur later and their final proxies remain much "
            "closer to the floors.",
            "",
            "## 4. Frozen budget decision",
            "",
            "**Decision: do not launch a larger-budget continuation under the "
            "same frozen configurations for the current paper experiment.**",
            "",
            "The paired low-epsilon comparison is limited by NOG-ZO, and the "
            "reserved seeds show that simply continuing the same NOG-ZO "
            "configuration is unlikely to recover lower thresholds. Extending "
            "only the baselines cannot restore paired observations. A larger "
            "budget would therefore add cost without addressing the limiting "
            "mechanism.",
            "",
            "The existing 20-seed formal result remains valid on its complete-"
            "pair interval and supports the qualitative depth-ratio claim. "
            "The smaller-epsilon region remains right-censored and should be "
            "reported as such.",
            "",
            "Any future attempt to improve the NOG-ZO tail would require a "
            "separate stability or parameter-schedule study. Such a study "
            "would constitute a new experiment and must not replace or be "
            "merged with the frozen formal result.",
            "",
            "## 5. Reproduction",
            "",
            "    conda run -n NOG python -m "
            "src.distributed.zo_anomaly_replication_analysis",
            "",
            "Machine-readable values are in "
            "anomaly_replication_comparison.csv and "
            "anomaly_replication_comparison.json.",
            "",
        ]
    )
    REPORT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not DIAGNOSTIC_RAW.exists():
        raise FileNotFoundError(
            "Step ZO-6B results.csv does not exist; replication is incomplete."
        )
    formal_results = pd.read_csv(FORMAL_RAW)
    diagnostic_results = pd.read_csv(DIAGNOSTIC_RAW)
    audit = audit_diagnostic(diagnostic_results)
    if audit["status"] != "pass":
        raise RuntimeError("; ".join(audit["errors"]))
    formal_tasks = task_diagnostics(formal_results)
    anomaly_tasks = task_diagnostics(diagnostic_results)
    comparison = comparison_table(formal_tasks, anomaly_tasks)
    comparison.to_csv(CSV_OUTPUT, index=False)
    payload = {
        "step": "ZO-6C",
        "audit": audit,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "comparison": comparison.to_dict(orient="records"),
        "decision": (
            "do_not_extend_same_frozen_configurations_for_current_paper"
        ),
        "formal_results_unchanged": True,
        "merge_diagnostic_with_formal": False,
    }
    JSON_OUTPUT.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    make_figure(formal_tasks, anomaly_tasks)
    write_report(comparison, audit)
    print(comparison.to_string(index=False))
    print(f"saved={REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
