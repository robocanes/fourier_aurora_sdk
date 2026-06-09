# GR2 12-Joint Standard Walk Lively Config

This folder packages the sim-approved conservative tuning for the vendor GR2
`LowerBodyCpgTask` 12-joint standard walk.

## Files

- `config_gr2t2v2_mix_without_waist_lively.yaml`
  - Install target for the normal task path:
    `/opt/fftai/fourier_aurora/config/sys_config/gr2/LowerBodyCpgTask/`
- `config_gr2t2v2_mix_without_waist_lively_whole_body_task_copy.yaml`
  - Matching copy for systems that load:
    `/opt/fftai/fourier_aurora/config/sys_config/gr2/whole_body_task/lower_body_cpg_task/config/`

## Sim Result

Tested in Isaac Sim with AuroraCore using standard FSM state `3`
(`RLLocomotion`) and velocity source `2` (`Navigation`).

Passed:

- forward: `vx=0.15`, `0.25`, `0.35`, `0.45`
- backward: `vx=-0.15`, `-0.25`
- spot yaw: `yaw=+0.25`, `-0.25`
- curved walk: `vx=0.20` with `yaw=+0.25` and `yaw=-0.25`
- lab navigation sequence: straight, curve, stop, curve opposite, stop

## Install

On the robot or sim image, first back up the current selector:

```bash
cd /opt/fftai/fourier_aurora/config/sys_config/gr2/LowerBodyCpgTask
cp config.yaml config.yaml.backup_before_lively
```

Copy the config:

```bash
cp /path/to/config_gr2t2v2_mix_without_waist_lively.yaml .
printf 'config_name: "config_gr2t2v2_mix_without_waist_lively.yaml"\n' > config.yaml
```

If the whole-body task path is used, also install the matching copy:

```bash
cd /opt/fftai/fourier_aurora/config/sys_config/gr2/whole_body_task/lower_body_cpg_task/config
cp config.yaml config.yaml.backup_before_lively
cp /path/to/config_gr2t2v2_mix_without_waist_lively_whole_body_task_copy.yaml \
  config_gr2t2v2_mix_without_waist_lively.yaml
printf 'config_name: "config_gr2t2v2_mix_without_waist_lively.yaml"\n' > config.yaml
```

Restart AuroraCore after changing the selector.

## Rollback

```bash
cd /opt/fftai/fourier_aurora/config/sys_config/gr2/LowerBodyCpgTask
cp config.yaml.backup_before_lively config.yaml
```

Then restart AuroraCore.

## First Physical Test

Use the same conservative sequence that passed in sim:

1. FSM state `2` PD stand.
2. FSM state `3` RLLocomotion.
3. Velocity source `2`.
4. Start with `vx=0.15`, `vy=0.0`, `yaw=0.0`.
5. Stop by setting velocity to `0.0, 0.0, 0.0` and remain in FSM state `3`.
