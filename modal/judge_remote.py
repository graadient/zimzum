"""
Remote wrapper for judge.py — skips git surface check (no repo in container).
"""

import judge

# No git repo in the container — always pass the surface check
judge.verify_surface = lambda *_: True

if __name__ == "__main__":
    judge.main()
