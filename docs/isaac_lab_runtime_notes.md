# Isaac Lab Runtime Notes for TRACER

## Current status

TRACER high-level modules are pure PyTorch and can run in a normal Python environment.

Validated modules:

```text
TracerHighLevel
TracerHighLevelLoop
TracerStepAdapter
TracerMockEnv

The mock environment can also run through the Isaac Lab launcher:
cd ~/IsaacLab
PYTHONPATH=~/Tracer/TRACER:$PYTHONPATH ./isaaclab.sh -p -m isaaclab_tracer.scripts.check_tracer_mock_env

## Isaac Sim runtime-dependent modules

Some Isaac Lab modules require the Isaac Sim / Kit runtime before import:
pxr
omni.physics
isaacsim.core
isaaclab.envs
isaaclab.sim
isaaclab.assets
isaaclab.scene

Running:
./isaaclab.sh -p

selects the Isaac Lab Python environment, but it does not necessarily launch the full Isaac Sim runtime.

## Runtime probe

Use:
cd ~/IsaacLab
PYTHONPATH=~/Tracer/TRACER:$PYTHONPATH ./isaaclab.sh -p -m isaaclab_tracer.scripts.check_isaacsim_runtime_api --headless

This script first checks imports before launching Isaac Sim, then launches Isaac Sim through Isaac Lab AppLauncher, and checks the imports again.

## Interpretation

If pxr, omni.physics, and isaaclab.envs are available only after AppLauncher, then the installation is likely fine.

If they are still unavailable after AppLauncher, then Isaac Sim package installation or runtime path configuration needs to be repaired.

## Next step

Once runtime imports work, implement a TRACER Isaac Lab environment skeleton using either:
DirectRLEnv
or
ManagerBasedRLEnv

The current preference is DirectRLEnv, because TRACER has a custom high-level loop and custom step adapter.
