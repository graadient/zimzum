"""
Modal app for running autoresearch GPU training remotely.

Usage (from repo root):
    modal run modal/app.py                    # train + judge
    modal run modal/app.py::prepare_data      # data prep only
"""

import pathlib

import modal

app = modal.App("zimzum")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
AUTORESEARCH_DIR = REPO_ROOT / "autoresearch"
MODAL_DIR = REPO_ROOT / "modal"

# Persistent volume for data cache (~/.cache/autoresearch)
data_vol = modal.Volume.from_name("autoresearch-data", create_if_missing=True)

# Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.9.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "kernels>=0.11.7",
        "numpy>=2.2.6",
        "pyarrow>=21.0.0",
        "requests>=2.32.0",
        "rustbpe>=0.1.0",
        "tiktoken>=0.11.0",
    )
)

WORK_DIR = "/work"

# Mount autoresearch/ source into /work, plus judge.py and judge_remote.py
code_mount = modal.Mount.from_local_dir(
    str(AUTORESEARCH_DIR),
    remote_path=WORK_DIR,
    condition=lambda path: path.endswith((".py", ".toml")),
)
judge_mount = modal.Mount.from_local_file(
    str(REPO_ROOT / "judge.py"),
    remote_path=f"{WORK_DIR}/judge.py",
)
judge_remote_mount = modal.Mount.from_local_file(
    str(MODAL_DIR / "judge_remote.py"),
    remote_path=f"{WORK_DIR}/judge_remote.py",
)


@app.function(
    image=image,
    volumes={"/cache": data_vol},
    mounts=[code_mount],
    timeout=600,
    cpu=4,
)
def prepare_data(num_shards: int = 10):
    """Download data and train tokenizer (no GPU needed)."""
    import os, sys

    os.environ["HOME"] = "/cache"  # prepare.py caches in ~/.cache/autoresearch
    sys.path.insert(0, WORK_DIR)

    from prepare import download_data, train_tokenizer

    download_data(num_shards)
    train_tokenizer()
    data_vol.commit()
    return "prepare done"


@app.function(
    image=image,
    gpu="H100",
    volumes={"/cache": data_vol},
    mounts=[code_mount, judge_mount, judge_remote_mount],
    timeout=900,
)
def train_and_judge():
    """Run train.py then judge.py on GPU, return metrics."""
    import os, sys, json, subprocess

    os.environ["HOME"] = "/cache"
    os.chdir(WORK_DIR)

    # Training
    print("=== Training ===")
    r = subprocess.run(
        [sys.executable, "train.py"],
        cwd=WORK_DIR, capture_output=True, text=True, timeout=600,
    )
    print(r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout)
    if r.returncode != 0:
        print(f"TRAIN FAILED:\n{r.stderr[-2000:]}")
        return {"status": "train_crash", "stderr": r.stderr[-2000:]}

    # Judging (judge_remote.py wraps judge.py, skipping git surface check)
    print("\n=== Judging ===")
    r = subprocess.run(
        [sys.executable, "judge_remote.py"],
        cwd=WORK_DIR, capture_output=True, text=True, timeout=300,
    )
    print(r.stdout)
    if r.returncode != 0:
        print(f"JUDGE FAILED:\n{r.stderr[-2000:]}")

    # Return metrics
    metrics_path = os.path.join(WORK_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return {"status": "no_metrics"}


@app.local_entrypoint()
def main():
    """Default: prepare data, then train + judge."""
    import json

    print("Ensuring data is prepared...")
    prepare_data.remote(num_shards=10)

    print("Starting training on H100...")
    metrics = train_and_judge.remote()

    print("\n=== Results ===")
    print(json.dumps(metrics, indent=2))
    if metrics.get("val_bpb"):
        print(f"\nval_bpb: {metrics['val_bpb']}")
