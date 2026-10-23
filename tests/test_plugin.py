"""CPU tests of the LeRobot glue; the engine and the GPU are stand-ins."""

import contextlib
import sys
import types

import pytest
import torch

import lerobot_policy_proprio as lpp


class FakePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = torch.nn.Linear(4, 3)
        self.config = types.SimpleNamespace()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        return cls().eval()

    def predict_action_chunk(self, batch, noise=None, **guidance):
        return self.lin(batch["x"]) + 0.01 * torch.randn(batch["x"].shape[0], 3)  # samples its own noise

    def select_action(self, batch):
        return self.predict_action_chunk(batch)[:, :1]


class FakeEngine:
    """Proprio stand-in: 'replay' is the policy's own computation, optionally perturbed."""

    def __init__(self, policy, drift=0.0, fail=False, target="model.sample_actions"):
        self.policy, self.drift, self.fail, self.target = policy, drift, fail, target
        self.bypassed = False

    def warmup(self):
        if self.fail:
            raise RuntimeError("capture failed")
        orig = self.policy._proprio_hook.originals["predict_action_chunk"]
        engine = self

        with torch.inference_mode():  # like Proprio: buffers made in inference mode, written on every tick
            buf = torch.zeros(2, 4)

        def replay(batch, noise=None, **guidance):
            if not engine.bypassed:
                buf.copy_(batch["x"])  # refused outside inference mode
            out = orig(batch, noise, **guidance)
            return out if engine.bypassed else out + engine.drift

        self.policy._proprio_hook.originals["predict_action_chunk"] = replay
        torch.rand(5)  # warm-up consumes randomness; the plugin must restore it
        return {"auto_target": self.target}

    def graph_warmup_calls(self):
        return 1

    @contextlib.contextmanager
    def bypass(self):
        self.bypassed = True
        try:
            yield
        finally:
            self.bypassed = False


@pytest.fixture
def gpu(monkeypatch):
    """Pretend the policy is on CUDA; random state saved/restored on the CPU generator only."""
    monkeypatch.setattr(lpp, "_policy_device", lambda p: torch.device("cuda"))

    def save(device):
        state = torch.get_rng_state()
        return lambda: torch.set_rng_state(state)

    monkeypatch.setattr(lpp, "_save_rng", save)
    engines = []

    def make(drift=0.0, fail=False, target="model.sample_actions"):
        def factory(policy, path, device):
            engines.append(FakeEngine(policy, drift, fail, target))
            return engines[-1]

        monkeypatch.setattr(lpp, "_make_engine", factory)
        return engines

    return make


def batch():
    return {"x": torch.arange(8, dtype=torch.float32).reshape(2, 4)}


def test_identical_actions_switch_proprio_on_and_keep_the_random_stream(gpu):
    gpu(drift=0.0)
    p = lpp.attach(FakePolicy().eval(), "lerobot/smolvla_base")
    torch.manual_seed(1)
    a = p.select_action(batch())
    after = torch.rand(3)
    assert p._proprio_hook.state == "on"

    q = FakePolicy()  # the same run without the plugin
    torch.manual_seed(1)
    assert torch.equal(a, q.select_action(batch()))
    assert torch.equal(after, torch.rand(3))


def test_differing_actions_fall_back_to_the_original(gpu):
    gpu(drift=1e-3)
    p = lpp.attach(FakePolicy().eval(), "x")
    torch.manual_seed(2)
    a = p.select_action(batch())
    assert p._proprio_hook.state == "eager"
    q = FakePolicy()
    torch.manual_seed(2)
    assert torch.equal(a, q.select_action(batch()))  # later ticks run the original path


@pytest.mark.parametrize("kind", ["fail", "no_target"])
def test_failures_leave_the_policy_unchanged(gpu, kind):
    gpu(fail=kind == "fail", target=None if kind == "no_target" else "t")
    p = lpp.attach(FakePolicy().eval(), "x")
    torch.manual_seed(3)
    a = p.predict_action_chunk(batch())
    assert p._proprio_hook.state == "eager"
    q = FakePolicy()
    torch.manual_seed(3)
    assert torch.equal(a, q.predict_action_chunk(batch()))


def test_cpu_policy_is_left_alone(monkeypatch):
    monkeypatch.setattr(lpp, "_make_engine", lambda *a: pytest.fail("no engine on CPU"))
    p = lpp.attach(FakePolicy().eval(), "x")
    p.select_action(batch())
    assert p._proprio_hook.state == "off"


def test_training_mode_is_not_prepared(gpu):
    engines = gpu()
    p = lpp.attach(FakePolicy().train(), "x")
    p.predict_action_chunk(batch())
    assert p._proprio_hook.state == "new" and not engines


def test_install_wraps_get_policy_class_everywhere_it_is_bound(monkeypatch):
    from lerobot.policies import factory

    original = factory.get_policy_class
    original = getattr(original, "__wrapped__", original)
    monkeypatch.setattr(factory, "get_policy_class", original)
    importer = types.ModuleType("fake_lerobot_entry")
    importer.get_policy_class = original  # like `from lerobot.policies import get_policy_class`
    monkeypatch.setitem(sys.modules, "fake_lerobot_entry", importer)
    monkeypatch.setattr(factory, "_get_policy_cls_from_policy_name", lambda name: FakePolicy)

    assert lpp.install()
    assert factory.get_policy_class is importer.get_policy_class is not original
    cls = importer.get_policy_class("smolvla")
    assert issubclass(cls, FakePolicy) and cls.__name__ == "FakePolicy"
    policy = cls.from_pretrained("lerobot/smolvla_base")
    assert isinstance(policy, FakePolicy) and policy._proprio_hook.path == "lerobot/smolvla_base"
    assert lpp.install()  # idempotent
    monkeypatch.setattr(factory, "get_policy_class", original)


def test_off_switch_and_training_command(monkeypatch):
    monkeypatch.setenv("PROPRIO_ENGINE", "off")
    assert not lpp.install()
    cls = lpp._accelerated(FakePolicy)
    assert getattr(cls.from_pretrained("x"), "_proprio_hook", None) is None
    monkeypatch.setenv("PROPRIO_ENGINE", "on")
    monkeypatch.setitem(sys.modules, "lerobot.scripts.lerobot_train", types.ModuleType("t"))
    assert getattr(cls.from_pretrained("x"), "_proprio_hook", None) is None


def test_lerobot_discovers_the_plugin_by_distribution_name():
    from importlib.metadata import distribution

    assert distribution("lerobot_policy_proprio").metadata["Name"].startswith("lerobot_policy_")


def test_callers_outside_inference_mode_get_an_editable_tensor(gpu):
    """LeRobot's async policy server calls the policy outside inference mode."""
    gpu(drift=0.0)
    p = lpp.attach(FakePolicy().eval(), "x")
    with torch.inference_mode():
        p.predict_action_chunk(batch())  # prepared
    assert p._proprio_hook.state == "on"
    with torch.no_grad():
        out = p.predict_action_chunk(batch())
    out.mul_(2)  # LeRobot post-processing may edit the chunk in place


def test_calls_with_unchecked_arguments_run_the_original_path(gpu):
    """LeRobot's real-time chunking passes guidance arguments the preparation never checked."""
    engines = gpu(drift=0.0)
    p = lpp.attach(FakePolicy().eval(), "x")
    with torch.inference_mode():
        p.predict_action_chunk(batch())
    assert p._proprio_hook.state == "on"
    engines[0].drift = 1.0  # anything not run through bypass would now differ
    q = FakePolicy()
    torch.manual_seed(5)
    with torch.inference_mode():
        a = p.predict_action_chunk(batch(), inference_delay=3)
    torch.manual_seed(5)
    assert torch.equal(a, q.predict_action_chunk(batch(), inference_delay=3))
