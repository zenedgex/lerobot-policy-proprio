"""Proprio for LeRobot: faster policy ticks, bit-identical actions.

LeRobot imports every installed package named ``lerobot_policy_*`` when its commands start
(``lerobot-rollout``, ``lerobot-record``, ``lerobot-eval``, the async robot client). Importing this
one makes every pretrained policy those commands load run through Proprio:

* before the first action, Proprio prepares the policy and checks, on the robot's own first
  observation, that its actions are bit-identical to the policy's original execution;
* if they are, every later tick uses Proprio; if they are not, the policy keeps running exactly as
  LeRobot runs it, and the log says why;
* random-number state is restored after the check, so a run with Proprio produces the same
  actions as the same run without it.

Nothing is changed for training, on machines without CUDA, or with ``PROPRIO_ENGINE=off``.
The engine itself is the ``proprio`` package (https://github.com/zenedgex/proprio-releases).
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import time

__version__ = "0.1.1"

log = logging.getLogger("proprio")

_OFF_VALUES = ("0", "off", "false", "no")


def enabled() -> bool:
    return os.environ.get("PROPRIO_ENGINE", "on").strip().lower() not in _OFF_VALUES


def _training() -> bool:
    """LeRobot's training command imports its module; never touch a policy being trained."""
    return "lerobot.scripts.lerobot_train" in sys.modules


# ----------------------------------------------------------------------------------- per-policy
_CHECKED_KWARGS = {"batch", "noise"}


class _Hook:
    """State of one loaded policy: new -> preparing -> on | eager."""

    def __init__(self, policy, path: str | None):
        self.policy = policy
        self.path = path
        self.state = "new"
        self.engine = None
        self.originals = {}

    def call(self, name: str, *args, **kwargs):
        orig = self.originals[name]
        if self.state == "new" and not self.policy.training:
            self.prepare(args[0] if args else kwargs.get("batch"))
        # Calls carrying arguments the preparation did not check (for example real-time chunking
        # guidance) run exactly as LeRobot runs them.
        unchecked = set(kwargs) - _CHECKED_KWARGS or len(args) > 1
        if (self.state == "eager" or (self.state == "on" and unchecked)) and self.engine is not None:
            with self.engine.bypass():
                return orig(*args, **kwargs)
        if self.state == "on":
            import torch

            if not torch.is_inference_mode_enabled():
                # Proprio's prepared tick runs in inference mode; callers outside it (LeRobot's async
                # policy server) get an ordinary tensor back that they may modify.
                with torch.inference_mode():
                    out = orig(*args, **kwargs)
                return out.clone() if isinstance(out, torch.Tensor) else out
        return orig(*args, **kwargs)

    def prepare(self, batch) -> None:
        self.state = "preparing"  # calls made while preparing go straight to the policy
        device = _policy_device(self.policy)
        if device is None or device.type != "cuda":
            log.info("proprio: policy is on %s; Proprio runs on NVIDIA GPUs, leaving it unchanged", device)
            self.state = "off"
            return
        restore_rng = _save_rng(device)
        t0 = time.perf_counter()
        try:
            self.engine = _make_engine(self.policy, self.path, device)
            manifest = self.engine.warmup() or {}
            target = manifest.get("compiled") or manifest.get("auto_target")
            if not target:
                log.warning("proprio: no part of this policy could be prepared exactly; running it unchanged")
                self.state = "eager"
                return
            same, eager_ms, fast_ms = self._check(batch)
            if same:
                self.state = "on"
                log.info("proprio: actions bit-identical to the original on this robot's first observation; "
                         "tick %.1f ms (original %.1f ms); prepared in %.0f s",
                         fast_ms, eager_ms, time.perf_counter() - t0)
            else:
                self.state = "eager"
                log.warning("proprio: actions differ from the original on this robot's observation; "
                            "running the policy unchanged (please report: hello@zenedgex.in)")
        except Exception:  # any failure leaves the policy exactly as LeRobot runs it
            log.exception("proprio: could not prepare this policy; running it unchanged")
            self.state = "eager" if self.engine is not None else "off"
        finally:
            restore_rng()

    def _check(self, batch) -> tuple[bool, float, float]:
        """Original execution and Proprio on the same observation and seed, bit for bit."""
        import torch

        predict = self.originals["predict_action_chunk"]

        def run(seed: int):
            torch.manual_seed(seed)
            _sync()
            t = time.perf_counter()
            with torch.inference_mode():
                out = predict(batch).clone()
            _sync()
            return out, (time.perf_counter() - t) * 1e3

        # A new observation shape is prepared on its first calls; run until it is in use.
        for i in range(self.engine.graph_warmup_calls() + 2):
            run(10_000 + i)
        fast, fast_ms = run(4242)
        with self.engine.bypass():
            ref, eager_ms = run(4242)
        return bool(torch.equal(fast, ref)), eager_ms, fast_ms


def _save_rng(device):
    """Snapshot CPU and device random state; returns the function that puts it back."""
    import torch

    cpu = torch.get_rng_state()
    dev = torch.cuda.get_rng_state(device)

    def restore():
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(dev, device)

    return restore


def _sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _policy_device(policy):
    try:
        return next(policy.parameters()).device
    except StopIteration:
        return None


def _make_engine(policy, path: str | None, device):
    """The Proprio engine around an already loaded LeRobot policy (kept separate for tests)."""
    from lerobot.policies import make_pre_post_processors
    from proprio.engine import PolicyEngine

    overrides = {"device_processor": {"device": str(device)}}
    pre, post = make_pre_post_processors(policy.config, pretrained_path=path, preprocessor_overrides=overrides,
                                         postprocessor_overrides=overrides)
    # Good citizen inside LeRobot's process: leave its thread count and garbage collector alone.
    return PolicyEngine(policy, pre, post, name=path or type(policy).__name__, config=policy.config,
                        mode="exact", device="cuda", pin_threads=None, gc_freeze=False)


def attach(policy, path: str | None = None):
    """Route a loaded LeRobot policy through Proprio (prepared lazily, on its first inference)."""
    if getattr(policy, "_proprio_hook", None) is not None:
        return policy
    hook = _Hook(policy, path)
    for name in ("select_action", "predict_action_chunk"):
        orig = getattr(policy, name, None)
        if orig is None:
            continue
        hook.originals[name] = orig
        object.__setattr__(policy, name, functools.partial(hook.call, name))
    object.__setattr__(policy, "_proprio_hook", hook)
    return policy


# ------------------------------------------------------------------------------- LeRobot hook
_CLASSES: dict[type, type] = {}


def _accelerated(cls: type) -> type:
    """A subclass of a LeRobot policy class whose ``from_pretrained`` attaches Proprio."""
    if getattr(cls, "_proprio_class", False) or not hasattr(cls, "from_pretrained"):
        return cls
    if cls in _CLASSES:
        return _CLASSES[cls]

    def from_pretrained(klass, *args, **kwargs):
        policy = super(sub, klass).from_pretrained(*args, **kwargs)
        if enabled() and not _training():
            path = kwargs.get("pretrained_name_or_path", args[0] if args else None)
            attach(policy, str(path) if path is not None else None)
        return policy

    sub = type(cls.__name__, (cls,), {"_proprio_class": True, "from_pretrained": classmethod(from_pretrained),
                                       "__module__": cls.__module__, "__qualname__": cls.__qualname__})
    _CLASSES[cls] = sub
    return sub


def install() -> bool:
    """Wrap LeRobot's ``get_policy_class`` wherever it is bound. Idempotent."""
    if not enabled():
        log.info("proprio: PROPRIO_ENGINE=off, LeRobot policies run unchanged")
        return False
    try:
        from lerobot.policies import factory
    except ImportError:
        return False
    original = getattr(factory.get_policy_class, "__wrapped__", None) or factory.get_policy_class
    if getattr(factory.get_policy_class, "_proprio", False):
        return True

    @functools.wraps(original)
    def get_policy_class(name: str):
        return _accelerated(original(name))

    get_policy_class._proprio = True
    # Modules that imported the name (`from lerobot.policies import get_policy_class`) hold their own
    # binding; rebind it in every loaded module so all LeRobot entry points see the same function.
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if isinstance(namespace, dict) and namespace.get("get_policy_class") is original:
            namespace["get_policy_class"] = get_policy_class
    return True


try:
    import proprio as _proprio  # noqa: F401
except ImportError:
    log.warning("proprio: the engine package is not installed; LeRobot policies run unchanged "
                "(install it from https://github.com/zenedgex/proprio-releases)")
else:
    install()
