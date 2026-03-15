"""Shared config and utilities for the zimzum harness."""

import os
import subprocess

import yaml

DEFAULT_PROJECT = "projects/gpt_pretrain/project.yaml"


def resolve_project_path(path=None):
    """Resolve the project.yaml path from arg, env var, or default."""
    return path or os.environ.get("ZIMZUM_PROJECT", DEFAULT_PROJECT)


def load_project(path=None):
    """Load project.yaml from the given path or ZIMZUM_PROJECT env var."""
    path = resolve_project_path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_yaml_path"] = path
    return cfg


def git(*args):
    """Run a git command and return stdout, or None on failure."""
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None
