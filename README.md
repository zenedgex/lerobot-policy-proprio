# Proprio for LeRobot

Install one package and LeRobot runs your policy faster, with **actions bit-identical to the
original**. No code changes, no new commands, no robot-side changes.

```sh
pip install lerobot_policy_proprio     # brings the Proprio engine on Linux
```

Then run LeRobot as you already do:

```sh
lerobot-rollout --policy.path=lerobot/smolvla_base ...
lerobot-record  --policy.path=your-org/your-policy ...
lerobot-eval    --policy.path=lerobot/pi05_base ...
```

LeRobot loads `lerobot_policy_*` packages automatically. Before the first action, Proprio prepares
the policy and checks, on your robot's own first observation, that its actions are bit-identical
to the policy's original execution. If they are, every later tick is faster. If they are not, your
policy keeps running exactly as LeRobot runs it, and the log says why. The log line looks like:

```
proprio: actions bit-identical to the original on this robot's first observation; tick 36.2 ms (original 246.1 ms); prepared in 7 s
```

## What to expect

Measured on an RTX 4090 inside LeRobot 0.6.1 itself, with and without this plugin, same observations
and seeds, 8 CPU threads for both:

| Policy | LeRobot path | Identical actions | LeRobot alone | With Proprio |
|---|---|---|---|---|
| lerobot/smolvla_base | rollout / record / eval loading | 20/20 chunks, 120/120 control steps | 268 ms | **36 ms (7.4x)** |
| lerobot/smolvla_base | LeRobot's async policy server | 20/20 chunks | 332 ms | **55 ms (6.1x)** |
| lerobot/pi05_base (float32, as LeRobot loads it) | rollout / record / eval loading | 20/20 chunks, 120/120 control steps | 356 ms | **202 ms (1.8x)** |
| lerobot/pi05_base (float32) | LeRobot's async policy server | 20/20 chunks | 394 ms | **224 ms (1.8x)** |

Preparing the policy takes a few seconds once, before the first action (7 s for SmolVLA, 12 s
for pi0.5 on the 4090). Certificates for these and other policies (GR00T N1.7, OpenVLA, H100):
https://huggingface.co/datasets/ZenEdgeX/proprio-certificates

## Details

- **Free for one robot**, no key needed. Fleets (many robots per GPU), signed certificates and
  support: hello@zenedgex.in
- **Requirements:** Linux, an NVIDIA GPU, LeRobot 0.6, Python 3.10–3.12. On other machines the
  plugin leaves LeRobot unchanged.
- **Turn it off** for one run: `PROPRIO_ENGINE=off lerobot-rollout ...`
- **Training is never touched.** The plugin only acts on policies loaded for inference.
- **Unverified call types run unchanged.** Calls with extra arguments (for example real-time chunking
  guidance) run exactly as LeRobot runs them.
- **Same random stream:** a run with Proprio produces the same actions as the same run without it.
- **Async inference:** LeRobot's policy server does not load plugins yet (a change has been
  proposed upstream). Until then, use `proprio lerobot-server` (same flags) on the GPU machine.
  Guide: https://github.com/zenedgex/proprio-releases/blob/main/LEROBOT.md

This plugin is MIT-licensed. The Proprio engine it calls is separate, proprietary software; patent
pending.
