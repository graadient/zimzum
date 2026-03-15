"""Edit surface verification. Checks that only allowed files were modified."""

import subprocess


def verify_surface(mutable_files):
    """Check that only files in mutable_files were modified between HEAD~1 and HEAD.
    Fail-closed: returns False if the check cannot be performed."""
    has_parent = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD~1"],
        capture_output=True,
    ).returncode == 0
    if not has_parent:
        print("NOTE: first commit (no parent) — surface check skipped.")
        return True

    try:
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().split("\n")
    except Exception:
        print("SURFACE CHECK FAILED: could not run git diff. Failing closed.")
        return False

    allowed = set(mutable_files)
    forbidden = [f for f in changed if f and f not in allowed]
    if forbidden:
        print(f"SURFACE VIOLATION: forbidden files modified: {forbidden}")
        return False
    return True
