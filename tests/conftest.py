import os
import sys

import pytest
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


@pytest.fixture
def skeleton_exp():
    cfg = OmegaConf.load(os.path.join(REPO, "configs", "experiment", "skeleton.yaml"))
    return OmegaConf.to_container(cfg.experiment, resolve=True)
