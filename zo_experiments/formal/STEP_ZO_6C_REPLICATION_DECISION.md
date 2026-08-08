# Step ZO-6C: replication result and budget decision

## 1. Integrity

The diagnostic audit passed 20/20 tasks on reserved seeds [200, 201, 202, 203, 204]. Each task reused the frozen formal candidate and ended at work 983,040. Diagnostic results remain separate from the 20 formal seeds.

## 2. Formal-versus-anomaly comparison

| method | confirmed floor: formal / anomaly | floor position: formal / anomaly | final/floor: formal / anomaly | early rebound: formal / anomaly | pattern replicated |
|---|---:|---:|---:|---:|---|
| NOG-ZO | 0.01493 / 0.01463 | 25.4% / 25.4% | 1.97 / 1.92 | 100% / 100% | yes |
| ME-DOL-ZO | 0.01271 / 0.01293 | 73.3% / 59.4% | 1.12 / 1.13 | 0% / 0% | yes |
| DGFM | 0.01051 / 0.01040 | 77.0% / 79.0% | 1.11 / 1.07 | 0% / 0% | yes |
| DGFM+ | 0.02676 / 0.02648 | 71.4% / 60.4% | 1.08 / 1.11 | 0% / 0% | yes |

![Formal and anomaly comparison](figures/anomaly_replication_comparison.png)

## 3. Replication conclusion

The NOG-ZO anomaly replicates exactly under the preregistered rule: all 20 formal seeds and all five reserved anomaly seeds reach their confirmed floor before half of the work budget and finish at more than 1.5 times that floor. The anomaly-seed mean floor position and final/floor ratio are close to the formal values. This is configuration-level late-run instability rather than an idiosyncrasy of the original formal seeds.

ME-DOL-ZO, DGFM, and DGFM+ replicate the contrasting pattern: no formal or anomaly seed satisfies the early-rebound definition. Their floors occur later and their final proxies remain much closer to the floors.

## 4. Frozen budget decision

**Decision: do not launch a larger-budget continuation under the same frozen configurations for the current paper experiment.**

The paired low-epsilon comparison is limited by NOG-ZO, and the reserved seeds show that simply continuing the same NOG-ZO configuration is unlikely to recover lower thresholds. Extending only the baselines cannot restore paired observations. A larger budget would therefore add cost without addressing the limiting mechanism.

The existing 20-seed formal result remains valid on its complete-pair interval and supports the qualitative depth-ratio claim. The smaller-epsilon region remains right-censored and should be reported as such.

Any future attempt to improve the NOG-ZO tail would require a separate stability or parameter-schedule study. Such a study would constitute a new experiment and must not replace or be merged with the frozen formal result.

## 5. Reproduction

    conda run -n NOG python -m src.distributed.zo_anomaly_replication_analysis

Machine-readable values are in anomaly_replication_comparison.csv and anomaly_replication_comparison.json.
