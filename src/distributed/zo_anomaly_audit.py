"""Step ZO-6A: diagnose censoring, rebound, and checkpoint sensitivity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = (
    ROOT
    / "outputs/distributed_zo/zo_theory_validation/formal/fixed_work_983040"
    / "results.csv"
)
FORMAL = ROOT / "zo_experiments/formal"
AUDIT = FORMAL / "audit.json"
FIGURES = FORMAL / "figures"
TASK_OUTPUT = FORMAL / "anomaly_task_diagnostics.csv"
BOUNDARY_OUTPUT = FORMAL / "boundary_checkpoint_sensitivity.csv"
SUMMARY_OUTPUT = FORMAL / "anomaly_method_summary.csv"
JSON_OUTPUT = FORMAL / "anomaly_audit.json"
REPORT_OUTPUT = FORMAL / "STEP_ZO_6A_ANOMALY_AUDIT.md"

METHODS = ["NOG-ZO", "ME-DOL-ZO", "DGFM", "DGFM+"]
DENSE_EPSILONS = [
    0.035,
    0.0325,
    0.030,
    0.0295,
    0.029,
    0.0285,
    0.028,
    0.0275,
    0.027,
    0.0265,
    0.026,
    0.0255,
    0.025,
    0.0225,
    0.020,
    0.0195,
    0.019,
    0.0185,
    0.018,
    0.0175,
    0.017,
    0.0165,
    0.016,
    0.0155,
    0.015,
    0.0145,
    0.014,
    0.0135,
    0.013,
    0.0125,
    0.012,
    0.0115,
    0.011,
    0.01075,
    0.0105,
    0.01025,
    0.010,
    0.0095,
    0.009,
]


def task_diagnostics(results: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (method, seed), frame in results.groupby(
        ["method", "formal_seed"], sort=True
    ):
        ordered = frame.sort_values("depth").reset_index(drop=True)
        proxy = ordered["stat_proxy"].to_numpy(dtype=float)
        depth = ordered["depth"].to_numpy(dtype=float)
        work = ordered["total_work"].to_numpy(dtype=float)
        raw_index = int(np.argmin(proxy))
        pair_threshold = np.maximum(proxy[:-1], proxy[1:])
        confirmed_index = int(np.argmin(pair_threshold))
        tail_count = max(8, len(proxy) // 5)
        normalized_tail_work = work[-tail_count:] / work[-1]
        tail_slope = float(
            np.polyfit(normalized_tail_work, proxy[-tail_count:], 1)[0]
        )
        confirmed_floor = float(pair_threshold[confirmed_index])
        confirmed_fraction = float(work[confirmed_index] / work[-1])
        final_over_floor = float(proxy[-1] / confirmed_floor)
        early_rebound = bool(
            confirmed_fraction < 0.5 and final_over_floor > 1.5
        )
        late_floor = bool(confirmed_fraction >= 0.8)
        if early_rebound:
            outlook = "extension_unlikely_without_stability_change"
        elif late_floor or tail_slope < 0:
            outlook = "extension_plausible_but_unproven"
        else:
            outlook = "extension_uncertain"
        records.append(
            {
                "method": str(method),
                "formal_seed": int(seed),
                "checkpoints": int(len(ordered)),
                "final_depth": int(depth[-1]),
                "final_work": int(work[-1]),
                "median_depth_gap": float(np.median(np.diff(depth))),
                "max_depth_gap": float(np.max(np.diff(depth))),
                "median_work_gap": float(np.median(np.diff(work))),
                "max_work_gap": float(np.max(np.diff(work))),
                "raw_floor": float(proxy[raw_index]),
                "raw_floor_depth": int(depth[raw_index]),
                "raw_floor_work": int(work[raw_index]),
                "raw_floor_work_fraction": float(
                    work[raw_index] / work[-1]
                ),
                "confirmed_floor": confirmed_floor,
                "confirmed_floor_depth": int(depth[confirmed_index]),
                "confirmed_floor_work": int(work[confirmed_index]),
                "confirmed_floor_work_fraction": confirmed_fraction,
                "raw_to_confirmed_floor_gap": float(
                    confirmed_floor - proxy[raw_index]
                ),
                "final_proxy": float(proxy[-1]),
                "final_over_confirmed_floor": final_over_floor,
                "last_20_percent_proxy_slope": tail_slope,
                "tail_improving": bool(tail_slope < 0),
                "early_rebound": early_rebound,
                "late_floor": late_floor,
                "extension_outlook": outlook,
            }
        )
    return pd.DataFrame(records)


def boundary_table(tasks: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for method in METHODS:
        frame = tasks.loc[tasks["method"] == method]
        for epsilon in DENSE_EPSILONS:
            single = frame["raw_floor"] <= epsilon
            confirmed = frame["confirmed_floor"] <= epsilon
            records.append(
                {
                    "method": method,
                    "epsilon": float(epsilon),
                    "seeds": int(len(frame)),
                    "single_checkpoint_hits": int(single.sum()),
                    "confirmed_two_checkpoint_hits": int(confirmed.sum()),
                    "transient_only_hits": int((single & ~confirmed).sum()),
                    "single_hit_rate": float(single.mean()),
                    "confirmed_hit_rate": float(confirmed.mean()),
                }
            )
    return pd.DataFrame(records)


def method_summary(
    tasks: pd.DataFrame, boundary: pd.DataFrame
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for method in METHODS:
        frame = tasks.loc[tasks["method"] == method]
        bframe = boundary.loc[boundary["method"] == method]
        max_loss = int(bframe["transient_only_hits"].max())
        max_rows = bframe.loc[bframe["transient_only_hits"] == max_loss]
        loss_epsilons = (
            ",".join(
                f"{value:.5f}"
                for value in max_rows["epsilon"].astype(float).tolist()
            )
            if max_loss > 0
            else "none"
        )
        records.append(
            {
                "method": method,
                "seeds": int(len(frame)),
                "checkpoints_per_seed": int(frame["checkpoints"].iloc[0]),
                "final_depth": int(frame["final_depth"].iloc[0]),
                "final_work": int(frame["final_work"].iloc[0]),
                "mean_raw_floor": float(frame["raw_floor"].mean()),
                "mean_confirmed_floor": float(
                    frame["confirmed_floor"].mean()
                ),
                "min_confirmed_floor": float(
                    frame["confirmed_floor"].min()
                ),
                "max_confirmed_floor": float(
                    frame["confirmed_floor"].max()
                ),
                "median_confirmed_floor_work_fraction": float(
                    frame["confirmed_floor_work_fraction"].median()
                ),
                "min_confirmed_floor_work_fraction": float(
                    frame["confirmed_floor_work_fraction"].min()
                ),
                "max_confirmed_floor_work_fraction": float(
                    frame["confirmed_floor_work_fraction"].max()
                ),
                "mean_final_over_confirmed_floor": float(
                    frame["final_over_confirmed_floor"].mean()
                ),
                "early_rebound_seeds": int(frame["early_rebound"].sum()),
                "late_floor_seeds": int(frame["late_floor"].sum()),
                "tail_improving_seeds": int(frame["tail_improving"].sum()),
                "max_hits_lost_to_confirmation": max_loss,
                "epsilon_at_max_confirmation_loss": loss_epsilons,
                "dominant_extension_outlook": str(
                    frame["extension_outlook"].mode().iloc[0]
                ),
            }
        )
    return pd.DataFrame(records)


def make_figures(
    results: pd.DataFrame, boundary: pd.DataFrame
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {
        "NOG-ZO": "#1f77b4",
        "ME-DOL-ZO": "#d62728",
        "DGFM": "#2ca02c",
        "DGFM+": "#9467bd",
    }
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True)
    for method, axis in zip(METHODS, axes.ravel()):
        method_frame = results.loc[results["method"] == method]
        interpolated: list[np.ndarray] = []
        grid = np.linspace(0.0, 1.0, 241)
        for _, frame in method_frame.groupby("formal_seed"):
            ordered = frame.sort_values("total_work")
            x = (
                ordered["total_work"].to_numpy(dtype=float)
                / float(ordered["total_work"].iloc[-1])
            )
            y = ordered["stat_proxy"].to_numpy(dtype=float)
            axis.plot(x, y, color=colors[method], alpha=0.14, linewidth=0.7)
            interpolated.append(np.interp(grid, x, y))
        mean_curve = np.mean(np.vstack(interpolated), axis=0)
        axis.plot(
            grid,
            mean_curve,
            color="black",
            linewidth=2.0,
            label="seed mean",
        )
        axis.set_title(method)
        axis.set_ylabel("stationarity proxy")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("fraction of fixed work budget")
    figure.suptitle(
        "Formal ZO trajectories: thin lines are the 20 formal seeds"
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "anomaly_proxy_trajectories.png", dpi=220)
    figure.savefig(FIGURES / "anomaly_proxy_trajectories.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True)
    for method, axis in zip(METHODS, axes.ravel()):
        frame = boundary.loc[boundary["method"] == method].sort_values(
            "epsilon", ascending=False
        )
        axis.plot(
            frame["epsilon"],
            frame["single_hit_rate"],
            linestyle="--",
            marker=".",
            label="one checkpoint",
            color="#ff7f0e",
        )
        axis.plot(
            frame["epsilon"],
            frame["confirmed_hit_rate"],
            linestyle="-",
            marker=".",
            label="two checkpoints",
            color=colors[method],
        )
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(method)
        axis.set_ylabel("hit rate")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("epsilon (smaller to the right)")
    figure.suptitle("Checkpoint-confirmation sensitivity near censoring")
    figure.tight_layout()
    figure.savefig(
        FIGURES / "boundary_checkpoint_sensitivity.png", dpi=220
    )
    figure.savefig(FIGURES / "boundary_checkpoint_sensitivity.pdf")
    plt.close(figure)


def selected_boundary_rows(boundary: pd.DataFrame) -> pd.DataFrame:
    selected = [
        0.030,
        0.029,
        0.028,
        0.027,
        0.026,
        0.025,
        0.020,
        0.019,
        0.018,
        0.017,
        0.0165,
        0.016,
        0.0155,
        0.015,
        0.0145,
        0.014,
        0.0135,
        0.013,
        0.012,
        0.011,
        0.0105,
        0.010,
    ]
    return boundary.loc[
        boundary["epsilon"].apply(
            lambda value: any(np.isclose(value, target) for target in selected)
        )
    ]


def write_report(
    summary: pd.DataFrame,
    boundary: pd.DataFrame,
    audit: dict[str, Any],
) -> None:
    lines = [
        "# Step ZO-6A: anomaly and censoring audit",
        "",
        "This audit reuses the frozen Step ZO-5A trajectories. It does not "
        "change parameters, discard seeds, or launch a new experiment.",
        "",
        f"Input integrity: {audit['tasks_observed']}/"
        f"{audit['tasks_expected']} tasks passed the Step ZO-5B audit.",
        "",
        "## 1. Diagnostic definitions",
        "",
        "- Raw floor: the minimum proxy at any single checkpoint.",
        "- Confirmed floor: the minimum, over adjacent checkpoint pairs, of "
        "the larger proxy in the pair. This is the smallest epsilon that can "
        "satisfy the preregistered two-consecutive-checkpoint rule.",
        "- Early rebound: the confirmed floor occurs before half of the work "
        "budget and the final proxy exceeds it by more than 50 percent.",
        "- Late floor: the confirmed floor occurs in the final 20 percent of "
        "the budget.",
        "- Extension outlook is a trajectory diagnostic, not a guarantee of "
        "what an extended run would do.",
        "",
        "## 2. Method-level diagnosis",
        "",
        "| method | confirmed floor mean [min, max] | median floor position | "
        "final/floor | early rebound | late floor | tail improving | "
        "extension outlook |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['method']} | {row['mean_confirmed_floor']:.5f} "
            f"[{row['min_confirmed_floor']:.5f}, "
            f"{row['max_confirmed_floor']:.5f}] | "
            f"{100 * row['median_confirmed_floor_work_fraction']:.1f}% | "
            f"{row['mean_final_over_confirmed_floor']:.2f} | "
            f"{int(row['early_rebound_seeds'])}/20 | "
            f"{int(row['late_floor_seeds'])}/20 | "
            f"{int(row['tail_improving_seeds'])}/20 | "
            f"{row['dominant_extension_outlook']} |"
        )
    lines.extend(
        [
            "",
            "NOG-ZO is the clearest anomaly: its confirmed floor occurs near "
            "one quarter of the work budget for every seed, after which the "
            "proxy rebounds substantially. Merely appending more iterations "
            "to the same fixed configuration is therefore not supported as "
            "a likely cure by the existing trajectories.",
            "",
            "The three baselines typically attain their floors later and end "
            "much closer to those floors. A larger budget may help some late-"
            "floor seeds, but this remains unproven and should be tested on "
            "reserved anomaly seeds before altering any formal claim.",
            "",
            "## 3. Checkpoint-confirmation sensitivity",
            "",
            "Each cell is one-checkpoint hits / confirmed two-checkpoint hits "
            "out of 20. A gap identifies transient crossings suppressed by "
            "the confirmation rule.",
            "",
            "| epsilon | NOG-ZO | ME-DOL-ZO | DGFM | DGFM+ |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    selected = selected_boundary_rows(boundary)
    for epsilon in sorted(selected["epsilon"].unique(), reverse=True):
        cells = []
        for method in METHODS:
            row = selected.loc[
                (selected["method"] == method)
                & np.isclose(selected["epsilon"], epsilon)
            ].iloc[0]
            cells.append(
                f"{int(row['single_checkpoint_hits'])}/"
                f"{int(row['confirmed_two_checkpoint_hits'])}"
            )
        lines.append(f"| {epsilon:.5f} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "The confirmation rule removes isolated noisy crossings, but it "
            "does not create the main low-epsilon failure: the confirmed-floor "
            "distribution and the late trajectory behavior remain the primary "
            "constraints.",
            "",
            "## 4. Figures",
            "",
            "![Proxy trajectories](figures/anomaly_proxy_trajectories.png)",
            "",
            "![Checkpoint sensitivity](figures/boundary_checkpoint_sensitivity.png)",
            "",
            "## 5. Step ZO-6B recommendation",
            "",
            "1. Keep the frozen formal results unchanged.",
            "2. Run the same frozen configurations on reserved anomaly seeds "
            "200 through 204 at the current budget to test whether the rebound "
            "and floor positions replicate.",
            "3. For NOG-ZO, do not assume a simple budget extension will help; "
            "first confirm the early-rebound pattern on anomaly seeds.",
            "4. For ME-DOL-ZO, DGFM, and DGFM+, an explicitly separate two-"
            "times-budget sensitivity run can be considered after replication.",
            "5. Label every Step ZO-6B run diagnostic. It must not be merged "
            "with the 20-seed formal curves or used to retune them.",
            "",
            "## 6. Reproduction",
            "",
            "    conda run -n NOG python -m src.distributed.zo_anomaly_audit",
            "",
            "Machine-readable outputs are anomaly_task_diagnostics.csv, "
            "boundary_checkpoint_sensitivity.csv, anomaly_method_summary.csv, "
            "and anomaly_audit.json.",
            "",
        ]
    )
    REPORT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise RuntimeError("Step ZO-5B audit must pass before ZO-6A.")
    results = pd.read_csv(RAW)
    tasks = task_diagnostics(results)
    boundary = boundary_table(tasks)
    summary = method_summary(tasks, boundary)
    tasks.to_csv(TASK_OUTPUT, index=False)
    boundary.to_csv(BOUNDARY_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    payload = {
        "step": "ZO-6A",
        "formal_results_sha256": audit["raw_results_sha256"],
        "classification": {
            "early_rebound": (
                "confirmed floor before 50% work and final/floor > 1.5"
            ),
            "late_floor": "confirmed floor at or after 80% work",
            "tail_window": "last 20% of checkpoints",
        },
        "method_summary": summary.to_dict(orient="records"),
    }
    JSON_OUTPUT.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    make_figures(results, boundary)
    write_report(summary, boundary, audit)
    print(summary.to_string(index=False))
    print(f"saved={REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
