# Phase-B Objective Selector β v0

- Rollouts: `173`
- Input dim: `9`
- Best epoch: `37`
- Best val KL: `0.008794`

β = [β_v, β_s, β_e]

## By world

| world | beta_v | beta_s | beta_e |
|---|---:|---:|---:|
| earth | 0.594 | 0.226 | 0.180 |
| stairs_single | 0.471 | 0.382 | 0.147 |
| tracer_sponge_firm_flat | 0.172 | 0.753 | 0.075 |

## By world::action

| world::action | beta_v | beta_s | beta_e |
|---|---:|---:|---:|
| earth::trot_cautious | 0.594 | 0.226 | 0.180 |
| earth::trot_mid | 0.594 | 0.226 | 0.180 |
| earth::trot_soft_mid_clear | 0.594 | 0.226 | 0.180 |
| earth::trot_solid_fast | 0.594 | 0.226 | 0.180 |
| stairs_single::trot_cautious | 0.471 | 0.382 | 0.147 |
| stairs_single::trot_mid | 0.471 | 0.382 | 0.147 |
| stairs_single::trot_soft_mid_clear | 0.471 | 0.382 | 0.147 |
| stairs_single::trot_solid_fast | 0.471 | 0.382 | 0.147 |
| tracer_sponge_firm_flat::sponge_mid_brake_clear | 0.172 | 0.753 | 0.075 |
| tracer_sponge_firm_flat::sponge_probe_crawlish | 0.172 | 0.753 | 0.075 |
| tracer_sponge_firm_flat::sponge_reach_then_brake | 0.172 | 0.753 | 0.075 |
| tracer_sponge_firm_flat::sponge_short_step_stable | 0.172 | 0.753 | 0.075 |
| tracer_sponge_firm_flat::sponge_slow_high_clear | 0.172 | 0.753 | 0.075 |
| tracer_sponge_firm_flat::trot_cautious | 0.172 | 0.753 | 0.075 |
| tracer_sponge_firm_flat::trot_mid | 0.172 | 0.753 | 0.075 |
| tracer_sponge_firm_flat::trot_soft_mid_clear | 0.172 | 0.753 | 0.075 |
| tracer_sponge_firm_flat::trot_solid_fast | 0.172 | 0.753 | 0.075 |
