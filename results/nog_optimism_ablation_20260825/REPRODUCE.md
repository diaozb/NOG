# Reproduce

所有命令均使用 CPU 环境，不依赖 `conda activate`：

```bash
/root/miniconda3/envs/NOG/bin/python scripts/run_nog_optimism_ablation.py pilot \
  --concurrency 4 --output results/nog_optimism_ablation_20260825

/root/miniconda3/envs/NOG/bin/python scripts/run_nog_optimism_ablation.py formal \
  --concurrency 4 --output results/nog_optimism_ablation_20260825

/root/miniconda3/envs/NOG/bin/python scripts/analyze_nog_optimism_ablation.py
```

配置文件为 `configs/distributed_cpu_fo_nog_optimism_ablation_v4.yaml`。formal 任务支持断点恢复；重跑 formal 命令会复用已经完成且哈希一致的 partial，不会重复训练。

实现入口为 `src/distributed/cpu_fo_algorithms.py` 中的 `NOG-FO` 和 `NOG-FO-NONOPT`。后者只把更新式替换为普通 projected online gradient，保留相同的两次初始化 oracle 和所有 work/depth 计数。
