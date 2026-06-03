# AGENTS.md — SAC

## Overview

Search as Code (SaC) is a Python library and CLI tool for agentic research orchestration.
It uses an LLM (OpenAI-compatible API) to generate and execute Python code that performs
web searches, extracts structured data, and synthesizes findings within a sandboxed
execution environment.

## Commands

| Command | Description |
|---------|------------|
| `pytest` | Run test suite |
| `ruff format` | Format code |
| `mdformat` | Format markdown |
| `prospector --with-tool ruff --with-tool mypy --with-tool pylint src/` | Lint + type check (with blending) |
| `vulture --min-confidence 90 src/` | Dead/unused code detection |
| `lizard src/ --CCN=15` | Code complexity analysis |

## Development

```bash
# Setup
pip install -e ".[test]"

# Test
pytest

# Format
ruff format src/ tests/

# Format markdown
mdformat .

# Lint + type check
prospector --with-tool ruff --with-tool mypy --with-tool pylint src/

# Find unused code
vulture --min-confidence 90 src/

# Analyze code complexity
lizard src/ --min-cyclomatic-complexity 10
```

## Testing

Tests are in `tests/` and follow pytest conventions. Coverage target is 80%+.
The test suite covers all public API classes, edge cases, error paths, and
property-based tests for utility functions.

## Code Style

- Format: ruff format
- Lint + Type check: prospector (runs ruff check + mypy + pylint with blending)
- Docstrings: Google style

## Release

Use `tools/release.sh` to automate version bumps, builds, and GitHub releases:

```bash
./tools/release.sh          # bump patch (default)
./tools/release.sh minor
./tools/release.sh major
```

The script:

- Checks working tree is clean; warns if not on `master`/`main`
- Runs `bumpversion <part> --tag --verbose`
- Pushes commit + tags
- Builds the package
- Creates a GitHub release with auto-generated notes
