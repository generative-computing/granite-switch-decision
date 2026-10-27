# Contributing to granite-switch-decision

## Getting Started

1. Fork the repository
2. Clone your fork and install the development dependencies:

   ```bash
   git clone https://github.com/<your-username>/granite-switch-decision.git
   cd granite-switch-decision
   pip install -e ".[dev]"
   ```

   The unit and spec tests need no GPU. `tests/test_openai_spec.py` downloads OpenAI's
   OpenAPI schema on first run and skips with a message when offline. See
   [docs/TESTING.md](docs/TESTING.md) for what each test covers and how to run the
   live-server checks.

3. Create a feature branch and make your changes
4. Check formatting, lint and tests before opening a pull request:

   ```bash
   ruff format --check src tests scripts ci examples
   ruff check src tests scripts ci examples
   ty check src
   python3 ci/check_headers.py
   pytest
   ```

5. Submit a pull request

## Sign off your commits (DCO)

Every commit must carry a Developer Certificate of Origin sign-off. By adding a
sign-off you certify that you wrote the change (or otherwise have the right to submit
it) under the project's licence (the
[full DCO text](https://developercertificate.org/)). A sign-off is a trailer line at the
end of the commit message:

```
Signed-off-by: Jane Doe <jane.doe@example.com>
```

`git commit -s` appends it automatically, generated from your Git identity:

```bash
git config user.name "Jane Doe"
git config user.email "jane.doe@example.com"
```

Use your real name and a reachable email. PRs with unsigned commits are blocked by the
DCO check until every commit is signed off. To sign off the last commit, amend it:

```bash
git commit --amend -s --no-edit
```

To sign off several commits, use `git rebase --signoff <base>` and force-push.

The repo's pre-commit config has a local DCO check that runs at `commit-msg` time. Install it with
`pre-commit install --hook-type commit-msg`, and unsigned commits then fail before they are pushed.

## Contribution Guidelines

- Keep changes surgical: touch only what the change needs, and match the existing
  style even where you would do it differently
- New behaviour needs a test. [docs/TESTING.md](docs/TESTING.md) records what each
  existing test covers
- Wire-format behaviour is pinned against the published OpenAI and TypeSafe schemas;
  if you change a route's shape, update the spec tests and the README's list of
  unsupported requests together

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before
participating.

## Questions

Open an issue.
