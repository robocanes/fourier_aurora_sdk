# GR2 12-Joint Standard Walk Fluid2 Config

This folder packages the current preferred config-only tuning for the vendor
GR2 `LowerBodyCpgTask` 12-joint standard walk.

It reuses the vendor policy:

```yaml
policy_path: "policy/gr2t2v2_mix_without_waist_new20.pt"
```

No policy retraining is included here. The changes are only YAML wrapper/CPG/PD
tuning around the original 12-joint policy.

## Files

- `config_gr2t2v2_mix_without_waist_fluid2.yaml`
  - Install target for:
    `/opt/fftai/fourier_aurora/config/sys_config/gr2t2v2/LowerBodyCpgTask/`
    or the equivalent `gr2/LowerBodyCpgTask/` sim path.
- `config_gr2t2v2_mix_without_waist_fluid2_whole_body_task_copy.yaml`
  - Matching copy for systems that load:
    `/opt/fftai/fourier_aurora/config/sys_config/gr2t2v2/whole_body_task/lower_body_cpg_task/config/`

## Difference From Lively

Compared to the earlier `standard_walk_12j_lively` package, `fluid2` is a more
aggressive sim-tested style variant:

- lower leg `kp_joint_space`
- lower leg `kd_joint_space`
- ankle pitch action scale increased from `2.0` to `2.25`
- ankle roll action scale increased from `1.0` to `1.10`
- cadence increased to `step_period: 0.78`, `step_freq: 1.28`
- walk contact ratio shortened to `[0.55, 0.55]`

## Install

Back up the selector:

```bash
cd /opt/fftai/fourier_aurora/config/sys_config/gr2t2v2/LowerBodyCpgTask
cp config.yaml config.yaml.backup_before_fluid2
```

Copy the config into that folder, then select it:

```bash
printf 'config_name: "config_gr2t2v2_mix_without_waist_fluid2.yaml"\n' > config.yaml
cat config.yaml
```

Restart AuroraCore after changing the selector.

## First Physical Test

Use a conservative sequence:

1. FSM state `2` PD stand.
2. FSM state `3` RLLocomotion.
3. Velocity source `2`.
4. Start with `vx=0.10` or `0.15`, `vy=0.0`, `yaw=0.0`.
5. Stop with velocity `0.0, 0.0, 0.0` and remain in FSM state `3`.

Do not start physical testing with `vx=0.25` or yaw commands. This config has
softer gains and larger ankle scaling than `lively`.
