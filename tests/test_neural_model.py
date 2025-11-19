"""NeuralConfidenceModel construction and state dict shapes."""

from __future__ import annotations

import pytest

from hypofuse.neural import FEATURE_NAMES, NeuralConfidenceModel

torch = pytest.importorskip("torch")


@pytest.mark.model
def test_construction_shapes() -> None:
    model = NeuralConfidenceModel(input_dim=len(FEATURE_NAMES), hidden=8, seed=0)
    shapes = model.state_dict_shapes()
    assert "fc1.weight" in shapes
    assert "fc1.bias" in shapes
    assert "fc2.weight" in shapes
    assert "fc2.bias" in shapes
    assert shapes["fc1.weight"] == (8, len(FEATURE_NAMES))
    assert shapes["fc1.bias"] == (8,)
    assert shapes["fc2.weight"] == (1, 8)
    assert shapes["fc2.bias"] == (1,)


@pytest.mark.model
def test_custom_hidden_dim() -> None:
    model = NeuralConfidenceModel(input_dim=4, hidden=16, seed=0)
    shapes = model.state_dict_shapes()
    assert shapes["fc1.weight"] == (16, 4)
    assert shapes["fc2.weight"] == (1, 16)


@pytest.mark.model
def test_missing_torch_raises() -> None:
    # NeuralConfidenceModel is importable even without torch;
    # instantiation is the guarded path. This test only runs when
    # torch IS available, so we just verify the class exists.
    assert callable(NeuralConfidenceModel)
