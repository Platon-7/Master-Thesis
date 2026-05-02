"""Smoke tests for the failure-rehearsal KL anchor (losses.md §Loss 3).

Tests pure-logic pieces — no GPU, no real model, no HF Trainer integration. Verifies:

  1. FailureKLBuffer push/peek/consume semantics, including the peek-when-below-N
     retention rule.
  2. detect_batch_quality on every input shape we expect (lists, mixed, missing).
  3. CORN Bernoulli KL: zero when distributions match, positive otherwise, correct
     gradient flow (through new only, never through old).
  4. Categorical (C51) KL: same correctness checks at K=10.
  5. Mask handling — both with and without per-frame masks, including the [B, T, 1]
     trailing-singleton form the trainer produces.
  6. Buffer entry construction: correct CPU placement, detachment, missing-key tolerance.
  7. End-to-end behavior of the buffer-policy state machine across an alternating
     failure/success sequence.

Run from anywhere:
    cd <thesis_root>
    /home/<user>/.conda/envs/robometer_gpu/bin/python \
        Robometer-LoRA/scripts/smoke_test_failure_kl.py

Exit code 0 = all tests pass. Non-zero = at least one assertion failed (stdout shows
which one). Designed to run in <2 seconds.
"""

from __future__ import annotations

import math
import os
import sys
import traceback
from typing import Callable, List

import torch
import torch.nn.functional as F

# Load the failure_kl module directly from its file rather than via the
# robometer.trainers package, whose __init__ pulls in unsloth/HF Trainer (needs a GPU
# and slows test startup). The module itself only depends on torch.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_THESIS_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_FAILURE_KL_PATH = os.path.join(
    _THESIS_ROOT, "Robometer", "robometer", "trainers", "failure_kl.py"
)
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("failure_kl", _FAILURE_KL_PATH)
_failure_kl = importlib.util.module_from_spec(_spec)
# Register before exec so @dataclass can resolve the module via sys.modules.
sys.modules["failure_kl"] = _failure_kl
_spec.loader.exec_module(_failure_kl)

FailureKLBuffer = _failure_kl.FailureKLBuffer
FailureKLEntry = _failure_kl.FailureKLEntry
build_buffer_entry = _failure_kl.build_buffer_entry
compute_failure_kl = _failure_kl.compute_failure_kl
detect_batch_quality = _failure_kl.detect_batch_quality
move_entry_to_device = _failure_kl.move_entry_to_device
build_concat_batch = _failure_kl.build_concat_batch
split_concat_logits = _failure_kl.split_concat_logits


# ---------------------------------------------------------------------------------------
# tiny test runner — prints pass/fail per test, exits non-zero on any failure
# ---------------------------------------------------------------------------------------

_PASSED: List[str] = []
_FAILED: List[str] = []


def _run(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        _PASSED.append(name)
        print(f"  PASS  {name}")
    except AssertionError as e:
        _FAILED.append(name)
        print(f"  FAIL  {name}\n        {e}")
    except Exception:
        _FAILED.append(name)
        print(f"  FAIL  {name}  (unexpected exception)")
        traceback.print_exc()


# ---------------------------------------------------------------------------------------
# Test: detect_batch_quality
# ---------------------------------------------------------------------------------------

def test_detect_quality_all_failure():
    assert detect_batch_quality(["failure"] * 8) == "failure"
    assert detect_batch_quality(["failed"] * 4) == "failure"
    assert detect_batch_quality(["suboptimal"] * 4) == "failure"
    assert detect_batch_quality(("failure",)) == "failure"


def test_detect_quality_all_success():
    assert detect_batch_quality(["successful"] * 8) == "successful"
    assert detect_batch_quality(["success"] * 4) == "successful"


def test_detect_quality_mixed():
    # Mixed batch should return None and warn — verified via return value (warning is logged)
    assert detect_batch_quality(["failure", "successful"]) is None


def test_detect_quality_empty_or_missing():
    assert detect_batch_quality(None) is None
    assert detect_batch_quality([]) is None
    assert detect_batch_quality(["unknown"]) is None  # unrecognized label


# ---------------------------------------------------------------------------------------
# Test: FailureKLBuffer push/peek/consume + retention rule
# ---------------------------------------------------------------------------------------

def _mk_entry(tag: int) -> FailureKLEntry:
    """Make a tiny entry with logits encoding the tag (so we can verify identity)."""
    return FailureKLEntry(
        inputs={"input_ids": torch.tensor([[tag]])},
        logits=torch.tensor([float(tag)]),
    )


def test_buffer_init_validation():
    try:
        FailureKLBuffer(0)
    except ValueError:
        return
    raise AssertionError("FailureKLBuffer(0) should have raised ValueError")


def test_buffer_push_peek_basic():
    buf = FailureKLBuffer(maxlen=3)
    assert len(buf) == 0
    assert buf.peek_head() is None
    assert buf.consume_head() is None  # empty: no error, returns None
    buf.push(_mk_entry(1))
    buf.push(_mk_entry(2))
    assert len(buf) == 2
    assert buf.peek_head().logits.item() == 1.0  # FIFO: head is oldest
    assert not buf.is_full
    buf.push(_mk_entry(3))
    assert buf.is_full  # exactly N=3 → full


def test_buffer_overflow_evicts_oldest():
    buf = FailureKLBuffer(maxlen=2)
    buf.push(_mk_entry(1))
    buf.push(_mk_entry(2))
    buf.push(_mk_entry(3))  # overflow: evict 1
    assert len(buf) == 2
    assert buf.peek_head().logits.item() == 2.0


def test_buffer_consume_below_capacity_does_not_pop():
    """The peek-when-below-N retention rule: consume on a not-full buffer keeps the
    entry alive so it can re-anchor multiple consecutive success steps."""
    buf = FailureKLBuffer(maxlen=3)
    buf.push(_mk_entry(1))
    buf.push(_mk_entry(2))  # buffer at 2/3, NOT full
    e = buf.consume_head()
    assert e.logits.item() == 1.0
    # Same entry should still be there.
    assert len(buf) == 2
    assert buf.peek_head().logits.item() == 1.0
    # Repeated consumes return the same head until buffer fills.
    e2 = buf.consume_head()
    assert e2.logits.item() == 1.0
    assert len(buf) == 2


def test_buffer_consume_at_capacity_pops_oldest():
    buf = FailureKLBuffer(maxlen=3)
    for tag in (1, 2, 3):
        buf.push(_mk_entry(tag))
    assert buf.is_full
    e = buf.consume_head()
    assert e.logits.item() == 1.0
    assert len(buf) == 2  # popped after consume because we WERE full
    # Next consume on the now non-full buffer becomes peek-only again.
    e2 = buf.consume_head()
    assert e2.logits.item() == 2.0
    assert len(buf) == 2


def test_buffer_alternating_failure_success_streams():
    """Walk a realistic alternating stream and verify the retention rule keeps the
    buffer at steady state without draining."""
    N = 4
    buf = FailureKLBuffer(maxlen=N)
    # Initial fill: 4 failures with no success in between → buffer at capacity.
    for tag in range(1, 5):
        buf.push(_mk_entry(tag))
    assert buf.is_full and len(buf) == N
    # Now alternate success/failure for a while; buffer should stay at exactly N.
    next_tag = 5
    for _ in range(20):
        e = buf.consume_head()  # success step
        assert e is not None
        assert len(buf) == N - 1  # popped because buffer WAS full
        buf.push(_mk_entry(next_tag))  # failure step refills
        next_tag += 1
        assert buf.is_full


# ---------------------------------------------------------------------------------------
# Test: KL math correctness — CORN (4 thresholds, Bernoulli per-threshold)
# ---------------------------------------------------------------------------------------

def test_corn_kl_zero_when_identical():
    """KL(P || P) = 0 exactly."""
    z = torch.randn(2, 16, 4)
    kl = compute_failure_kl(z.clone(), z.clone())
    assert torch.allclose(kl, torch.tensor(0.0), atol=1e-6), f"expected 0, got {kl.item()}"


def test_corn_kl_positive_when_different():
    """KL > 0 when distributions differ."""
    z_old = torch.randn(2, 16, 4)
    z_new = z_old + 0.5
    kl = compute_failure_kl(z_old, z_new)
    assert kl.item() > 0.0
    assert torch.isfinite(kl)


def test_corn_kl_grad_flows_only_through_new():
    """Gradient should flow through z_new but not through z_old."""
    z_old = torch.randn(1, 4, 4, requires_grad=True)
    z_new = torch.randn(1, 4, 4, requires_grad=True)
    kl = compute_failure_kl(z_old, z_new)
    kl.backward()
    assert z_new.grad is not None and z_new.grad.abs().sum().item() > 0.0
    # z_old MUST NOT have a non-trivial gradient — even if PyTorch sets it to 0 (because
    # we detached inside compute_failure_kl), the assert is that it's effectively zero.
    if z_old.grad is not None:
        assert z_old.grad.abs().sum().item() == 0.0, (
            f"z_old should have no gradient through KL; got {z_old.grad.abs().sum().item()}"
        )


def test_corn_kl_matches_manual_bernoulli():
    """For a single threshold, the per-frame KL must match the closed-form Bernoulli KL."""
    # Pick z values so σ(z_old) and σ(z_new) are easy to reason about.
    z_old = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])           # σ = 0.5 everywhere
    z_new = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])           # σ = 0.731 everywhere
    p_old = 0.5
    p_new = 1.0 / (1.0 + math.exp(-1.0))                      # ≈ 0.7311
    bernoulli_kl_per_thresh = (
        p_old * (math.log(p_old) - math.log(p_new))
        + (1 - p_old) * (math.log(1 - p_old) - math.log(1 - p_new))
    )
    expected = 4 * bernoulli_kl_per_thresh                    # summed across 4 thresholds
    kl = compute_failure_kl(z_old, z_new)
    assert math.isclose(kl.item(), expected, rel_tol=1e-4), (
        f"manual={expected:.6f}  computed={kl.item():.6f}"
    )


# ---------------------------------------------------------------------------------------
# Test: KL math correctness — categorical (C51 with K=10 bins)
# ---------------------------------------------------------------------------------------

def test_categorical_kl_zero_when_identical():
    z = torch.randn(2, 16, 10)
    kl = compute_failure_kl(z.clone(), z.clone())
    assert torch.allclose(kl, torch.tensor(0.0), atol=1e-6), f"expected 0, got {kl.item()}"


def test_categorical_kl_positive_when_different():
    z_old = torch.randn(2, 16, 10)
    z_new = z_old + 0.3 * torch.randn_like(z_old)
    kl = compute_failure_kl(z_old, z_new)
    assert kl.item() > 0.0
    assert torch.isfinite(kl)


def test_categorical_kl_matches_F_kl_div():
    """Cross-check against torch.nn.functional.kl_div for a known input."""
    z_old = torch.randn(1, 1, 10)
    z_new = torch.randn(1, 1, 10)
    p_old = F.softmax(z_old, dim=-1)
    log_p_new = F.log_softmax(z_new, dim=-1)
    # F.kl_div expects (log_q, p) and computes KL(p || q) when reduction='sum'.
    expected = F.kl_div(log_p_new, p_old, reduction="sum").item()
    actual = compute_failure_kl(z_old, z_new).item()
    assert math.isclose(actual, expected, rel_tol=1e-4), f"expected {expected}, got {actual}"


# ---------------------------------------------------------------------------------------
# Test: KL mask handling
# ---------------------------------------------------------------------------------------

def test_kl_with_per_frame_mask():
    """Mask=0 frames shouldn't contribute. Setting only a subset of frames to differ should
    give the same KL whether or not the matching frames are included (as long as the mask
    averages over only the differing frames)."""
    B, T, K = 1, 4, 4
    z_old = torch.zeros(B, T, K)
    z_new = z_old.clone()
    z_new[:, 0, :] = 2.0  # only frame 0 differs

    # If we mask out frame 0, KL should be 0 (only matching frames count).
    mask_skip_diff = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    kl = compute_failure_kl(z_old, z_new, mask=mask_skip_diff)
    assert torch.allclose(kl, torch.tensor(0.0), atol=1e-6), f"with diff masked out, KL should be 0, got {kl.item()}"

    # If we mask in only frame 0, KL is positive.
    mask_only_diff = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    kl = compute_failure_kl(z_old, z_new, mask=mask_only_diff)
    assert kl.item() > 0.0


def test_kl_with_trailing_singleton_mask():
    """target_progress_mask in the trainer is [B, T, 1]; the helper should auto-squeeze."""
    B, T, K = 1, 4, 4
    z_old = torch.zeros(B, T, K)
    z_new = torch.ones(B, T, K)
    mask_3d = torch.ones(B, T, 1)
    mask_2d = torch.ones(B, T)
    kl_3d = compute_failure_kl(z_old, z_new, mask=mask_3d).item()
    kl_2d = compute_failure_kl(z_old, z_new, mask=mask_2d).item()
    assert math.isclose(kl_3d, kl_2d, rel_tol=1e-6), f"squeeze should be transparent; {kl_3d} vs {kl_2d}"


# ---------------------------------------------------------------------------------------
# Test: build_buffer_entry / move_entry_to_device
# ---------------------------------------------------------------------------------------

def test_build_entry_detaches_and_cpus():
    progress_inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long).cuda() if torch.cuda.is_available() else torch.tensor([[1, 2, 3]], dtype=torch.long),
        "target_progress_mask": torch.ones(1, 16),
        "predict_last_frame_mask": torch.ones(1, 16),
        # An unrelated key that should be silently dropped (not in _BUFFERED_INPUT_KEYS):
        "irrelevant": torch.zeros(99, 99),
    }
    progress_pred = torch.randn(1, 16, 4, requires_grad=True)
    entry = build_buffer_entry(progress_inputs, progress_pred)
    # Logits and inputs all on CPU.
    assert entry.logits.device.type == "cpu"
    for k, v in entry.inputs.items():
        if isinstance(v, torch.Tensor):
            assert v.device.type == "cpu", f"{k} not on cpu"
    # Detached: requires_grad False, no grad_fn.
    assert not entry.logits.requires_grad
    # Irrelevant key not buffered.
    assert "irrelevant" not in entry.inputs
    # Required keys present.
    assert "input_ids" in entry.inputs
    assert "target_progress_mask" in entry.inputs


def test_move_entry_to_device_dtype():
    progress_inputs = {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}
    progress_pred = torch.randn(1, 16, 4)
    entry = build_buffer_entry(progress_inputs, progress_pred)
    inp_dev, logits_dev = move_entry_to_device(entry, torch.device("cpu"), dtype=torch.float64)
    # Input tensors keep their stored dtype (input_ids is long), logits are cast.
    assert inp_dev["input_ids"].dtype == torch.long
    assert logits_dev.dtype == torch.float64


# ---------------------------------------------------------------------------------------
# Test: build_concat_batch  (concat fast path)
# ---------------------------------------------------------------------------------------

def test_concat_basic_same_seq_len():
    """Equal-length inputs concat cleanly along dim 0."""
    succ = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "pixel_values": torch.randn(2 * 16, 3 * 14 * 14),  # 2 samples × 16 frames
        "image_grid_thw": torch.tensor([[1, 14, 14]] * 2),
        "demo_frames_count": torch.zeros(2, dtype=torch.int32),
    }
    fail = {
        "input_ids": torch.tensor([[7, 8, 9]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "pixel_values": torch.randn(1 * 16, 3 * 14 * 14),
        "image_grid_thw": torch.tensor([[1, 14, 14]]),
        "demo_frames_count": torch.zeros(1, dtype=torch.int32),
    }
    combined, b_succ, b_fail = build_concat_batch(succ, fail, pad_token_id=0)
    assert b_succ == 2 and b_fail == 1
    assert combined["input_ids"].shape == (3, 3)
    assert combined["pixel_values"].shape == (48, 3 * 14 * 14)
    assert combined["image_grid_thw"].shape == (3, 3)
    assert combined["demo_frames_count"].shape == (3,)
    # Order preserved: success first, failure second.
    assert torch.equal(combined["input_ids"][:2], succ["input_ids"])
    assert torch.equal(combined["input_ids"][2:], fail["input_ids"])


def test_concat_pads_shorter_sequence():
    """Different sequence lengths: shorter side gets padded with pad_token_id."""
    succ = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),     # S=4
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    fail = {
        "input_ids": torch.tensor([[5, 6]]),            # S=2 (shorter)
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }
    combined, b_succ, b_fail = build_concat_batch(succ, fail, pad_token_id=99)
    assert combined["input_ids"].shape == (2, 4)
    # Failure row got padded with pad_token_id=99 in positions 2 and 3.
    assert combined["input_ids"][1].tolist() == [5, 6, 99, 99]
    # attention_mask padding is always 0 (not pad_token_id).
    assert combined["attention_mask"][1].tolist() == [1, 1, 0, 0]


def test_concat_split_round_trips():
    """Output of build_concat_batch + split_concat_logits returns the original batch sizes."""
    succ = {
        "input_ids": torch.tensor([[1, 2, 3]] * 4),  # B=4
        "attention_mask": torch.ones(4, 3, dtype=torch.long),
    }
    fail = {
        "input_ids": torch.tensor([[7, 8, 9]] * 2),  # B=2
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
    }
    _, b_succ, b_fail = build_concat_batch(succ, fail)
    # Simulate the model's output: logits shape [B_succ + B_fail, T, K]
    logits = torch.randn(b_succ + b_fail, 16, 4)
    succ_out, fail_out = split_concat_logits(logits, b_succ)
    assert succ_out.shape == (4, 16, 4)
    assert fail_out.shape == (2, 16, 4)
    assert torch.equal(logits, torch.cat([succ_out, fail_out], dim=0))


def test_concat_one_sided_optional_field_passthrough():
    """If only one side has an optional field (e.g. demo_frames_count), it passes through
    rather than crashing."""
    succ = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "demo_frames_count": torch.tensor([16], dtype=torch.int32),
    }
    fail = {
        "input_ids": torch.tensor([[3, 4]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        # NB: no demo_frames_count — older buffered entry from before ICL.
    }
    combined, _, _ = build_concat_batch(succ, fail)
    assert "demo_frames_count" in combined
    # Should at least pass through the success-side tensor.
    assert torch.equal(combined["demo_frames_count"], succ["demo_frames_count"])


# ---------------------------------------------------------------------------------------
# Test: detach-backbone fast path  (head-only re-forward)
# ---------------------------------------------------------------------------------------

class _StubBackbone(torch.nn.Module):
    """Minimal stand-in for the backbone: maps input_ids → fake hidden state."""
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.embed = torch.nn.Embedding(20, hidden_dim)
    def forward(self, input_ids):
        return self.embed(input_ids).mean(dim=1, keepdim=True).expand(-1, 16, -1)


class _StubHead(torch.nn.Module):
    """Minimal CORN head: hidden_dim → 4 logits."""
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_dim, 4)
    def forward(self, x):
        return self.linear(x)


class _StubModel(torch.nn.Module):
    """Mimics the relevant subset of RBM: backbone + progress_head, with a forward that
    matches the head-input shape so the forward_pre_hook fires correctly."""
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.backbone = _StubBackbone(hidden_dim)
        self.progress_head = _StubHead(hidden_dim)
    def forward(self, input_ids):
        h = self.backbone(input_ids)            # [B, T=16, hidden_dim]
        return self.progress_head(h)            # [B, T=16, 4]


def test_detach_backbone_grad_flows_only_through_head():
    """Run a stub model: capture head input via forward_pre_hook in no_grad, re-run head
    with grad. Verify gradient lands ONLY on head params, NOT on backbone params."""
    torch.manual_seed(0)
    model = _StubModel()
    backbone_params_before = [p.detach().clone() for p in model.backbone.parameters()]

    captured = []
    def pre_hook(module, args):
        if args:
            captured.append(args[0])

    handle = model.progress_head.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            input_ids = torch.randint(0, 20, (2, 5))
            _ = model(input_ids)
    finally:
        handle.remove()

    assert captured, "forward_pre_hook should have captured one input"
    head_input = captured[0]
    assert not head_input.requires_grad  # was inside no_grad

    # Now re-run head with grad on the detached input.
    head_input_detached = head_input.detach()
    head_input_detached.requires_grad_(False)
    head_logits = model.progress_head(head_input_detached)
    fake_target = torch.randn_like(head_logits)
    loss = ((head_logits - fake_target) ** 2).mean()
    loss.backward()

    # Head params have grad; backbone params do NOT.
    for p in model.progress_head.parameters():
        assert p.grad is not None and p.grad.abs().sum().item() > 0.0, "head should have grad"
    for p in model.backbone.parameters():
        assert p.grad is None, "backbone must NOT receive gradient on head-only re-forward"


def test_detach_backbone_correctness_matches_full_forward():
    """Numerically: head_only_re_forward(x) should produce the same logits as a full
    forward (under the same dropout state). We disable dropout via eval() to make the
    comparison deterministic."""
    torch.manual_seed(1)
    model = _StubModel()
    model.eval()  # avoid any nondeterminism (none here, but explicit)
    input_ids = torch.randint(0, 20, (3, 4))

    with torch.no_grad():
        full = model(input_ids)

    captured = []
    def pre_hook(module, args):
        if args:
            captured.append(args[0])
    handle = model.progress_head.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            _ = model(input_ids)
    finally:
        handle.remove()
    head_only = model.progress_head(captured[0].detach())

    assert torch.allclose(full, head_only, atol=1e-6), (
        f"head-only re-forward diverged from full forward; max diff "
        f"{(full - head_only).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------------------

ALL_TESTS = [
    ("detect_quality.all_failure", test_detect_quality_all_failure),
    ("detect_quality.all_success", test_detect_quality_all_success),
    ("detect_quality.mixed_returns_None", test_detect_quality_mixed),
    ("detect_quality.empty_or_missing", test_detect_quality_empty_or_missing),
    ("buffer.init_validation", test_buffer_init_validation),
    ("buffer.push_peek_basic", test_buffer_push_peek_basic),
    ("buffer.overflow_evicts_oldest", test_buffer_overflow_evicts_oldest),
    ("buffer.consume_below_capacity_does_not_pop", test_buffer_consume_below_capacity_does_not_pop),
    ("buffer.consume_at_capacity_pops", test_buffer_consume_at_capacity_pops_oldest),
    ("buffer.alternating_stream_steady_state", test_buffer_alternating_failure_success_streams),
    ("corn_kl.zero_when_identical", test_corn_kl_zero_when_identical),
    ("corn_kl.positive_when_different", test_corn_kl_positive_when_different),
    ("corn_kl.grad_only_through_new", test_corn_kl_grad_flows_only_through_new),
    ("corn_kl.matches_manual_bernoulli", test_corn_kl_matches_manual_bernoulli),
    ("categorical_kl.zero_when_identical", test_categorical_kl_zero_when_identical),
    ("categorical_kl.positive_when_different", test_categorical_kl_positive_when_different),
    ("categorical_kl.matches_F_kl_div", test_categorical_kl_matches_F_kl_div),
    ("mask.per_frame_filters_correctly", test_kl_with_per_frame_mask),
    ("mask.trailing_singleton_squeeze", test_kl_with_trailing_singleton_mask),
    ("build_entry.detaches_and_cpus", test_build_entry_detaches_and_cpus),
    ("move_entry.dtype_cast", test_move_entry_to_device_dtype),
    ("concat.basic_same_seq_len", test_concat_basic_same_seq_len),
    ("concat.pads_shorter_sequence", test_concat_pads_shorter_sequence),
    ("concat.split_round_trips", test_concat_split_round_trips),
    ("concat.one_sided_optional_field_passthrough", test_concat_one_sided_optional_field_passthrough),
    ("detach_backbone.grad_only_through_head", test_detach_backbone_grad_flows_only_through_head),
    ("detach_backbone.matches_full_forward", test_detach_backbone_correctness_matches_full_forward),
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} smoke tests for failure-rehearsal KL anchor")
    print("-" * 70)
    for name, fn in ALL_TESTS:
        _run(name, fn)
    print("-" * 70)
    print(f"PASS: {len(_PASSED)} / {len(ALL_TESTS)}")
    if _FAILED:
        print(f"FAIL: {len(_FAILED)} — {', '.join(_FAILED)}")
        return 1
    print("All passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
