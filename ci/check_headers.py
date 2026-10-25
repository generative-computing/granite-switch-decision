#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check that every Python file under src/, tests/ and scripts/ carries an
SPDX-License-Identifier header in its first two lines (after an optional
shebang). Pre-commit hook and CI-able.

  python3 ci/check_headers.py            # check
  python3 ci/check_headers.py --fix      # insert the plain Apache-2.0 header

Files whose licence is a conjunction (the MIT-adapted ones, per
THIRD_PARTY_NOTICES.md) are listed in AND_MIT_FILES and must be fixed by hand:
the script never rewrites an existing header, it only inserts the default one.
"""

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_HEADER = "# SPDX-License-Identifier: Apache-2.0"
# These two carry MIT-adapted code (see THIRD_PARTY_NOTICES.md) and are headed
# "Apache-2.0 AND MIT"; the script checks for a header, never assumes which.
AND_MIT_FILES = {"src/granite_switch_decision/plugin.py", "tests/test_plugin.py"}


def has_header(path: pathlib.Path) -> bool:
    lines = path.read_text(encoding="utf-8").split("\n", 2)
    return any(line.startswith("# SPDX-License-Identifier:") for line in lines[:2])


def fix(path: pathlib.Path) -> None:
    text = path.read_text(encoding="utf-8")
    if text.startswith("#!"):
        first, rest = text.split("\n", 1)
        path.write_text(f"{first}\n{DEFAULT_HEADER}\n{rest}", encoding="utf-8")
    else:
        path.write_text(f"{DEFAULT_HEADER}\n{text}", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="insert the default header where missing")
    ap.add_argument("files", nargs="*", help="check these files (default: the src/tests/scripts globs)")
    args = ap.parse_args()

    if args.files:
        paths = [pathlib.Path(f).resolve() for f in args.files]
    else:
        paths = (
            sorted(ROOT.glob("src/*.py"))
            + sorted(ROOT.glob("src/*/*.py"))
            + sorted(ROOT.glob("tests/*.py"))
            + sorted(ROOT.glob("scripts/*.py"))
        )

    missing = []
    for path in paths:
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if not has_header(path):
            missing.append(rel)
            if args.fix:
                fix(path)
                print(f"fixed: {rel}")
    if missing and not args.fix:
        print("missing SPDX headers (run ci/check_headers.py --fix):")
        for rel in missing:
            print(f"  {rel}")
        if AND_MIT_FILES & set(missing):
            print("note: AND_MIT_FILES need 'Apache-2.0 AND MIT', not the default header")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
