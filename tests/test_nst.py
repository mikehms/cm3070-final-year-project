"""Unit tests: run with  python -m pytest tests -q"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nst.colour import recombine_luminance, rgb_to_yiq, yiq_to_rgb  # noqa: E402
from nst.losses import blended_gram_targets, gram_matrix, total_variation_loss  # noqa: E402
from nst.models import VGGExtractor, build_vgg_features  # noqa: E402


def test_gram_is_symmetric_and_psd():
    f = torch.rand(1, 8, 5, 7)
    g = gram_matrix(f)[0]
    assert torch.allclose(g, g.T, atol=1e-6)
    assert torch.linalg.eigvalsh(g).min() > -1e-6


def test_gram_is_translation_invariant():
    """Style statistics ignore spatial arrangement: a circular shift keeps the Gram matrix."""
    f = torch.rand(1, 4, 6, 6)
    assert torch.allclose(gram_matrix(f), gram_matrix(torch.roll(f, 2, dims=3)), atol=1e-6)


def test_tv_zero_for_flat_image():
    assert total_variation_loss(torch.full((1, 3, 8, 8), 0.3)).item() == 0.0


def test_layer_counts():
    _, n16 = build_vgg_features("vgg16")
    _, n19 = build_vgg_features("vgg19")
    assert sum(n.startswith("conv") and not n.endswith("_linear") for n in n16) == 13
    assert sum(n.startswith("conv") and not n.endswith("_linear") for n in n19) == 16


def test_extractor_outputs_requested_layers_and_is_frozen():
    net = VGGExtractor("vgg19", ["conv1_1", "conv4_2"])
    out = net(torch.rand(1, 3, 64, 64))
    assert set(out) == {"conv1_1", "conv4_2"}
    assert out["conv4_2"].shape[1] == 512
    assert not any(p.requires_grad for p in net.parameters())


def test_yiq_round_trip_and_colour_preservation():
    x = torch.rand(1, 3, 10, 10)
    assert torch.allclose(yiq_to_rgb(rgb_to_yiq(x)), x, atol=1e-5)
    grey = rgb_to_yiq(x)[:, :1].repeat(1, 3, 1, 1)
    assert torch.allclose(recombine_luminance(grey, x), x.clamp(0, 1), atol=1e-4)


def test_blend_of_one_style_equals_that_style():
    feats = {"a": torch.rand(1, 4, 5, 5)}
    g = blended_gram_targets([feats, {"a": torch.rand(1, 4, 5, 5)}], ["a"], [1.0, 0.0])
    assert torch.allclose(g["a"], gram_matrix(feats["a"]))
