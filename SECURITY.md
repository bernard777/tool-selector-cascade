# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | ✅ Yes             |
| < 0.1   | ❌ No              |

Only the **latest minor release** receives security fixes.
We strongly recommend always using the latest version.

---

## Reporting a Vulnerability

**Please do NOT open a public GitHub issue for security vulnerabilities.**

To report a vulnerability responsibly:

1. **Email:** ndongoambassa7@gmail.com  
   Subject: `[SECURITY] tool-selector-cascade — <short description>`

2. **GitHub Private Vulnerability Reporting** (preferred):  
   Use [GitHub's private security advisory](https://github.com/bernard777/tool-selector-cascade/security/advisories/new).

### What to include

- A clear description of the vulnerability.
- Steps to reproduce (minimal proof-of-concept if possible).
- Affected versions.
- Potential impact (data exposure, key leakage, denial-of-service, etc.).

---

## Response SLA

| Stage                            | Target time    |
|----------------------------------|----------------|
| Acknowledgement of your report   | 48 hours       |
| Initial triage and severity      | 5 business days|
| Fix released (critical/high)     | 14 days        |
| Fix released (medium/low)        | 30 days        |
| Public disclosure (after patch)  | 90 days        |

We follow coordinated disclosure: we ask that you keep the vulnerability
private until a patch is released (or until the 90-day deadline).

---

## Security Controls in This Library

### API Key Handling
- All API keys are read from **environment variables** (`ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY`, `GOOGLE_API_KEY`).
- API key patterns are **redacted** from exception messages before any logging
  via `_redact_secrets()` in `llm_picker.py`.

### Async Safety
- Concurrent access to the Google `genai` global state is protected by
  `asyncio.Lock` (`_google_lock`).

### Timeout Enforcement
- All external API calls enforce configurable timeouts via `asyncio.wait_for()`.
  Default: 30 seconds.

### Dependency Supply Chain
- All Git-sourced dependencies are pinned to a specific tag (`rev = "vX.Y.Z"`).
- Dependencies are audited weekly via `pip-audit` in the security workflow.
- CodeQL SAST scans run on every PR and weekly on the `master` branch.

---

## Credits

We thank all researchers who responsibly disclose security issues.
Credited reporters will be listed here and in the relevant CHANGELOG entry.
