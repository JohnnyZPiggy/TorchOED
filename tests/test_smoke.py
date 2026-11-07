import ast
from importlib import metadata, util
from pathlib import Path

import torchoed


def test_installed_package_without_pydoe():
    assert util.find_spec("pydoe") is None
    assert metadata.version("torchoed") == torchoed.__version__
    mandatory = [r for r in metadata.requires("torchoed") if "extra ==" not in r]
    assert mandatory == ["torch>=2.2"]


def test_package_imports_only_torch():
    for source in Path(torchoed.__file__).parent.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imports = {node.module.split(".")[0] for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom) and not node.level}
        imports |= {name.name.split(".")[0] for node in ast.walk(tree)
                    if isinstance(node, ast.Import) for name in node.names}
        assert imports <= {"torch"}
        assert ".numpy(" not in source.read_text() and ".cpu(" not in source.read_text()
