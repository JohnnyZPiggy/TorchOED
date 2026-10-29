import ast
from importlib import metadata, util
from pathlib import Path

import torch

import torchoed
from torchoed._smoke import _gram


def test_installed_package_without_pydoe():
    assert util.find_spec("pydoe") is None
    assert metadata.version("torchoed") == torchoed.__version__
    mandatory = [r for r in metadata.requires("torchoed") if "extra ==" not in r]
    assert mandatory == ["torch>=2.0"]
    assert not hasattr(torchoed, "optimal_design")


def test_deterministic_cpu_gram():
    features = torch.tensor([[1., -1.], [1., 0.], [1., 1.]], dtype=torch.float64)
    actual = _gram(features)
    expected = torch.tensor([[3., 0.], [0., 2.]], dtype=torch.float64)
    assert actual.shape == (2, 2)
    assert actual.dtype == features.dtype
    assert actual.device == features.device == torch.device("cpu")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(actual, actual.mT)
    assert bool(torch.all(torch.linalg.eigvalsh(actual) >= 0))


def test_private_smoke_has_no_hidden_regularization():
    features = torch.tensor([[1., 1.], [1., 1.]], dtype=torch.float64)
    expected = torch.full((2, 2), 2., dtype=torch.float64)
    torch.testing.assert_close(_gram(features), expected, atol=0, rtol=0)
    assert torch.linalg.det(_gram(features)) == 0


def test_package_imports_only_torch():
    for source in Path(torchoed.__file__).parent.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imports = {node.module.split(".")[0] for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom)}
        imports |= {name.name.split(".")[0] for node in ast.walk(tree)
                    if isinstance(node, ast.Import) for name in node.names}
        assert imports <= {"torch"}
        assert ".numpy(" not in source.read_text() and ".cpu(" not in source.read_text()
