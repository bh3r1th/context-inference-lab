import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def dataset(tmp_path):
    spec = importlib.util.spec_from_file_location("fixtures", Path(__file__).parents[1] /
                                                 "scripts/generate_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "data"
    module.generate(root, paragraphs=4)
    return root
