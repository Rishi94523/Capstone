"""
Tests for the proof-of-computation verifier.

Simulates an honest browser client (float32 layer math, like the JS shard
engine) and several classes of cheaters, and checks the verifier's verdicts.
"""

import numpy as np
import pytest

from app.ml.model_store import get_model_store
from app.ml.proof_verifier import (
    ProofVerifier,
    canonical_vector_hash,
    compute_proof_hash,
)


@pytest.fixture(scope="module")
def model():
    return get_model_store().get_default()


@pytest.fixture()
def verifier():
    # audit_rate=0 so tests exercise the projection path deterministically
    return ProofVerifier(audit_rate=0.0)


def client_compute(model, x, start, end):
    """
    Simulate the browser client: float32 weights, per-layer float32 storage
    of results (Float32Array), activations applied between layers.
    """
    h = np.asarray(x, dtype=np.float32)
    pre_activations = []
    for layer in model.layers[start:end]:
        z = (h.astype(np.float64) @ layer.weights.astype(np.float64)
             + layer.biases.astype(np.float64))
        z = z.astype(np.float32)  # Float32Array storage
        pre_activations.append([float(v) for v in z])
        h = model.apply_activation(z.astype(np.float64), layer.activation).astype(
            np.float32
        )
    return pre_activations


def build_proof(model, x, start, end, task_id="task-1", sample_id="sample-1",
                prediction_hash=""):
    pre_activations = client_compute(model, x, start, end)
    output_hashes = [canonical_vector_hash(z) for z in pre_activations]
    proof_hash = compute_proof_hash(
        task_id, sample_id, start, len(pre_activations), output_hashes, prediction_hash
    )
    return pre_activations, output_hashes, proof_hash


def random_input(seed=3):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, 784).astype(np.float32).tolist()


class TestHonestClient:
    def test_full_model_segment_passes(self, model, verifier):
        x = random_input()
        pre, hashes, proof_hash = build_proof(model, x, 0, 3)
        report = verifier.verify_segment(
            model, 0, x, pre, hashes, proof_hash, "task-1", "sample-1"
        )
        assert report.valid, report.reason
        assert report.predicted_label in model.labels
        assert report.probabilities is not None
        assert abs(float(report.probabilities.sum()) - 1.0) < 1e-3

    def test_single_layer_segment_passes(self, model, verifier):
        x = random_input()
        pre, hashes, proof_hash = build_proof(model, x, 0, 1)
        report = verifier.verify_segment(
            model, 0, x, pre, hashes, proof_hash, "task-1", "sample-1"
        )
        assert report.valid, report.reason
        assert report.predicted_label is None  # not final layer
        assert report.final_activation is not None
        assert len(report.final_activation) == 128

    def test_mid_pipeline_segment_passes(self, model, verifier):
        """Segment starting from a handed-over activation (distributed case)."""
        x = random_input()
        # First contributor computes layer 0
        pre0, h0, p0 = build_proof(model, x, 0, 1)
        report0 = verifier.verify_segment(model, 0, x, pre0, h0, p0, "task-1", "sample-1")
        assert report0.valid

        # Second contributor continues from the stored activation
        handoff = [float(v) for v in report0.final_activation]
        pre1, h1, p1 = build_proof(model, handoff, 1, 3, task_id="task-2")
        report1 = verifier.verify_segment(
            model, 1, handoff, pre1, h1, p1, "task-2", "sample-1"
        )
        assert report1.valid, report1.reason
        assert report1.predicted_label is not None

    def test_pieced_result_matches_direct_inference(self, model, verifier):
        """Distributed segments must piece together to the same label."""
        x = random_input(seed=11)
        direct = model.predict(np.asarray(x, dtype=np.float64))
        direct_label = model.labels[int(np.argmax(direct))]

        activation = x
        report = None
        for start in range(3):
            pre, hashes, ph = build_proof(
                model, activation, start, start + 1, task_id=f"t-{start}"
            )
            report = verifier.verify_segment(
                model, start, activation, pre, hashes, ph, f"t-{start}", "sample-1"
            )
            assert report.valid, report.reason
            activation = [float(v) for v in report.final_activation]

        assert report.predicted_label == direct_label


class TestCheaters:
    def test_fabricated_outputs_fail(self, model, verifier):
        """Client invents plausible-looking outputs without computing."""
        x = random_input()
        rng = np.random.default_rng(99)
        fake_pre = [[float(v) for v in rng.normal(0, 2, 128)]]
        hashes = [canonical_vector_hash(fake_pre[0])]
        proof_hash = compute_proof_hash("task-1", "sample-1", 0, 1, hashes, "")
        report = verifier.verify_segment(
            model, 0, x, fake_pre, hashes, proof_hash, "task-1", "sample-1"
        )
        assert not report.valid
        assert "projection" in report.reason

    def test_tampered_single_value_fails(self, model, verifier):
        """Honest computation with one perturbed output value."""
        x = random_input()
        pre, _, _ = build_proof(model, x, 0, 1)
        pre[0][37] += 0.5  # tamper
        hashes = [canonical_vector_hash(pre[0])]
        proof_hash = compute_proof_hash("task-1", "sample-1", 0, 1, hashes, "")
        report = verifier.verify_segment(
            model, 0, x, pre, hashes, proof_hash, "task-1", "sample-1"
        )
        assert not report.valid
        assert "projection" in report.reason

    def test_wrong_input_fails(self, model, verifier):
        """Client computed on a different input than assigned (precompute attack)."""
        x_assigned = random_input(seed=1)
        x_other = random_input(seed=2)
        pre, hashes, proof_hash = build_proof(model, x_other, 0, 1)
        report = verifier.verify_segment(
            model, 0, x_assigned, pre, hashes, proof_hash, "task-1", "sample-1"
        )
        assert not report.valid

    def test_replayed_proof_fails(self, model, verifier):
        """Proof generated for one task replayed against another."""
        x = random_input()
        pre, hashes, proof_hash = build_proof(model, x, 0, 1, task_id="task-A")
        report = verifier.verify_segment(
            model, 0, x, pre, hashes, proof_hash, "task-B", "sample-1"
        )
        assert not report.valid
        assert "proof hash" in report.reason

    def test_hash_data_mismatch_fails(self, model, verifier):
        """Submitted vectors don't match the committed hashes."""
        x = random_input()
        pre, hashes, proof_hash = build_proof(model, x, 0, 1)
        pre[0][0] += 1.0  # change data but keep old hashes
        report = verifier.verify_segment(
            model, 0, x, pre, hashes, proof_hash, "task-1", "sample-1"
        )
        assert not report.valid
        assert "commitment" in report.reason

    def test_wrong_size_fails(self, model, verifier):
        x = random_input()
        pre = [[0.0] * 64]  # wrong output size for layer 0
        hashes = [canonical_vector_hash(pre[0])]
        proof_hash = compute_proof_hash("task-1", "sample-1", 0, 1, hashes, "")
        report = verifier.verify_segment(
            model, 0, x, pre, hashes, proof_hash, "task-1", "sample-1"
        )
        assert not report.valid

    def test_spot_audit_catches_subtle_drift(self, model):
        """
        Small consistent perturbations below projection noise still get
        caught by the full-recompute audit.
        """
        auditing_verifier = ProofVerifier(audit_rate=1.0)
        x = random_input()
        pre, _, _ = build_proof(model, x, 0, 1)
        # nudge every element just above audit tolerance but below projection
        # noise threshold for a single value
        pre[0] = [v + 0.002 for v in pre[0]]
        hashes = [canonical_vector_hash(pre[0])]
        proof_hash = compute_proof_hash("task-1", "sample-1", 0, 1, hashes, "")
        report = auditing_verifier.verify_segment(
            model, 0, x, pre, hashes, proof_hash, "task-1", "sample-1"
        )
        assert not report.valid


class TestVerificationCost:
    def test_projection_check_is_cheaper_than_recompute(self, model):
        """
        The point of the design: verification work should be much smaller
        than the client's computation. Compare op counts.
        """
        client_ops = sum(
            l.input_size * l.output_size for l in model.layers
        )
        verify_ops = sum(
            (l.input_size + l.output_size) * 4  # NUM_PROJECTIONS
            for l in model.layers
        )
        assert verify_ops * 10 < client_ops  # >10x asymmetry even on a tiny model
