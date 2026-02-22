# Parkour Task

## Basic Usage Guidelines

### Parkour Task

**Task ID:** `Instinct-Parkour-Target-Amp-G1-v0`

1. Go to `config/g1/g1_parkour_target_amp_cfg.py` and set the `path` and `filtered_motion_selection_filepath` in `AmassMotionCfg` to the reference motion you want to use.

2. Train the policy:
```bash
instinct-train Instinct-Parkour-Target-Amp-G1-v0
```

3. Play trained policy (`--load-run` must be provided, absolute path is recommended, or use `--agent random` to visualize untrained policy):

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name>
```

4. Export trained policy (`--load-run` must be provided, absolute path is recommended):

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name> --export-onnx
```

## Common Options

- `--num-envs`: Number of parallel environments (default varies by task)
- `--load-run`: Run name to load checkpoint from for playing
- `--video`: Record training/playback videos
- `--export-onnx`: Export the trained model to ONNX format for onboard deployment during playing
