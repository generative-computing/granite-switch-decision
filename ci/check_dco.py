#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""commit-msg hook: every non-merge commit must carry a DCO sign-off trailer.

python3 ci/check_dco.py .git/COMMIT_EDITMSG
"""

import pathlib
import subprocess
import sys


def in_merge() -> bool:
    return subprocess.run(["git", "rev-parse", "-q", "--verify", "MERGE_HEAD"], capture_output=True).returncode == 0


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_dco.py <commit-message-file>", file=sys.stderr)
        return 2
    if in_merge():
        return 0
    message = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    if any(line.startswith("Signed-off-by: ") for line in message.splitlines()):
        return 0
    print("Missing Signed-off-by trailer. Use: git commit -s", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
