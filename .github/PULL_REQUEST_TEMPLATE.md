## Description

<!-- 
  What does this PR do? Why is this change needed?
  Link to the related issue: Closes #<issue-number>
-->

Closes #

## Type of Change

<!-- Check all that apply -->

- [ ] 🐛 Bug fix (non-breaking fix to an existing issue)
- [ ] ✨ New feature (non-breaking addition)
- [ ] 💥 Breaking change (fix or feature that changes existing behavior)
- [ ] 🔒 Security fix (addresses a vulnerability)
- [ ] ♻️ Refactor (no functional change)
- [ ] 📝 Documentation update
- [ ] ⚡ Performance improvement
- [ ] 🧪 Tests only

## Changes

<!-- Bullet-point list of specific changes. Be precise — the AI reviewer reads this. -->

- 
- 

## Testing

<!-- How did you test these changes? -->

- [ ] I ran `pytest tests/ -v` and all tests pass
- [ ] I added new tests for the changed behavior
- [ ] Coverage has not dropped below 80%
- [ ] I tested with `pytest --cov=src/tool_selector_cascade --cov-fail-under=80`

## Quality Checklist

- [ ] `ruff check .` passes
- [ ] `black --check .` passes
- [ ] `isort --check-only .` passes
- [ ] `mypy src/ --strict` passes (zero errors)

## Security Checklist (if applicable)

- [ ] No API keys or secrets are hardcoded
- [ ] New dependencies are pinned to an exact version or hash
- [ ] `pip-audit` shows no new vulnerabilities for added dependencies
- [ ] External API calls have timeout enforcement

## CHANGELOG

- [ ] `CHANGELOG.md` has been updated under `## [Unreleased]`

## Target Branch

<!-- PRs must target `dev`, not `master` (except release PRs from `dev` → `master`) -->

- [ ] This PR targets `dev` (standard feature/fix)
- [ ] This PR targets `master` (release PR from `dev` only)
