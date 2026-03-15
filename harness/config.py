"""Load project configuration from project.yaml."""

import os
import yaml


def load_project(path=None):
    """Load project.yaml from the given path or ZIMZUM_PROJECT env var."""
    if path is None:
        path = os.environ.get("ZIMZUM_PROJECT", "projects/gpt_pretrain/project.yaml")
    with open(path) as f:
        return yaml.safe_load(f)
