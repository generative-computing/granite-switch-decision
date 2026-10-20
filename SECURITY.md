# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in granite-switch-systemone, please report it
responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please report security issues by emailing the maintainers directly or through
GitHub's private vulnerability reporting feature.

## Supported Versions

Security updates are provided for the latest release only.

## Security Considerations

This plugin mounts two request routes (`/v1/systemone` and `/v1/decisions`) on a vLLM
server alongside `/v1/chat/completions`. When deploying it:

- **Authentication**: the plugin deliberately provides none of its own; use vLLM's
  `--api-key`, which covers these routes too, whenever the server is reachable by
  anyone you do not fully trust
- **Model access**: ensure proper access controls around the serving checkpoint and
  the machine it runs on
- **Caller text is untrusted input**: the plugin escapes it against prompt injection,
  but a decision adapter's probabilities are a prior, not an authorisation decision
- **Dependencies**: keep vLLM and the surrounding stack updated to receive security
  patches

## Responsible Disclosure

We appreciate your efforts to responsibly disclose your findings and will make every
effort to acknowledge your contributions.
