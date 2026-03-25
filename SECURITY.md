# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

**Email:** goering.jared@gmail.com

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

I'll respond within 48 hours and work with you on a fix before any public disclosure.

## Scope

Supermemory stores potentially sensitive data (memories extracted from conversations). Security concerns include:

- **Data leakage** through API endpoints
- **SQL injection** in search or query parameters
- **Unauthorized access** to the memory database
- **Sensitive data in logs** or error messages

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | Yes       |
| 0.1.x   | Yes       |

## Known Supply Chain Risks

**litellm (dependency):** On 2026-03-24, litellm versions 1.82.7 and 1.82.8 were compromised via a supply chain attack (TeamPCP/Trivy CI/CD compromise). These versions exfiltrate credentials, SSH keys, and environment variables. Our pyproject.toml explicitly excludes these versions. If you installed supermemory between 2026-03-24 10:52 UTC and the PyPI yank (~20:15 UTC), verify your litellm version: `pip show litellm`. Any version other than 1.82.7 or 1.82.8 is safe.

## Design Principles

- **Local-first:** Data stays on your machine by default
- **No telemetry:** Nothing phones home
- **API keys stay local:** Only used for LLM calls you configure
