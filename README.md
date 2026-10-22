# Proprio for LeRobot

Install one package and LeRobot runs your policy faster, with **actions bit-identical to the
original**. No code changes, no new commands, no robot-side changes.

```sh
pip install lerobot_policy_proprio
pip install proprio-0.1.2-cp312-cp312-manylinux*_x86_64.whl   # the engine, from https://github.com/zenedgex/proprio-releases
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
proprio: actions bit-identical to the original on this robot's first observation; tick 37.9 ms (original 241.3 ms)
```

## What to expect

Measured on an RTX 4090 with stock LeRobot checkpoints, model tick, all actions identical:

| Policy | Original | With Proprio |
|---|---|---|
| lerobot/smolvla_base | 232 ms | 37–40 ms |
| lerobot/pi05_base (bf16) | 257 ms | 97 ms |
| nvidia/GR00T-N1.7-3B | 170 ms | 64 ms |

The certificates behind these numbers: https://huggingface.co/datasets/ZenEdgeX/proprio-certificates

## Details

- **Free for one robot**, no key needed. Fleets (many robots per GPU), signed certificates and
  support: hello@zenedgex.in
- **Requirements:** Linux, an NVIDIA GPU, LeRobot 0.6, Python 3.10–3.12. On other machines the
  plugin leaves LeRobot unchanged.
- **Turn it off** for one run: `PROPRIO_ENGINE=off lerobot-rollout ...`
- **Training is never touched.** The plugin only acts on policies loaded for inference.
- **Same random stream:** a run with Proprio produces the same actions as the same run without it.
- **Async inference:** LeRobot's policy server does not load plugins yet; use
  `proprio lerobot-server` (same flags) on the GPU machine. Guide:
  https://github.com/zenedgex/proprio-releases/blob/main/LEROBOT.md

This plugin is MIT-licensed. The Proprio engine it calls is separate, proprietary software; patent
pending.
