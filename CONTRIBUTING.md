# Contributing to granite-switch-systemone

## Getting Started

1. Fork the repository
2. Clone your fork and install the development dependencies:

   ```bash
   git clone https://github.com/<your-username>/granite-switch-systemone.git
   cd granite-switch-systemone
   pip install -e ".[dev]"
   ```

   The unit and spec tests need no GPU. Two of them fetch pinned third-party material
   on first run (OpenAI's OpenAPI schema; TypeSafe's wire models come from the
   `typesafe-sdk` dev dependency) and skip with a message when offline. See
   [docs/TESTING.md](docs/TESTING.md) for what each test covers and how to run the
   live-server checks.

3. Create a feature branch and make your changes
4. Check formatting, lint and tests before opening a pull request:

   ```bash
   ruff format --check src tests scripts ci examples
   ruff check src tests scripts ci examples
   ty check src
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

Use your real name and a reachable email. If you forgot it on your last commit, amend it:

```bash
git commit --amend -s --no-edit
```

## Contribution Guidelines

- Keep changes surgical: touch only what the change needs, and match the existing
  style even where you would do it differently
- New behaviour needs a test. [docs/TESTING.md](docs/TESTING.md) records what each
  existing test covers
- Wire-format behaviour is pinned against the published OpenAI and TypeSafe schemas;
  if you change a route's shape, update the spec tests and the deliberate-exceptions
  list in the README together

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before
participating.

## Questions

Open an issue or start a discussion.
