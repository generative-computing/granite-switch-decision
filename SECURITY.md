# Security Policy

## Reporting a Vulnerability

Report vulnerabilities through GitHub's private vulnerability reporting (the
**Report a vulnerability** button under this repository's **Security** tab). Please do
not open a public issue.

## Supported Versions

Security updates are provided for the latest release only.

## Security Considerations

This plugin mounts two request routes (`/v1/systemone` and `/v1/decisions`) on a vLLM
server alongside `/v1/chat/completions`. When deploying it:

- **Authentication**: the plugin deliberately provides none of its own; use vLLM's
  `--api-key`, which covers these routes too, whenever the server is reachable by
  anyone you do not fully trust
- **Model access**: restrict who can read the checkpoint and log in to the serving machine
- **Caller text is untrusted input**: the plugin escapes it against prompt injection,
  but a decision adapter's probabilities are a prior, not an authorisation decision
- **Dependencies**: keep vLLM and the surrounding stack updated to receive security
  patches
