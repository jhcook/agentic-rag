Repository Rules

This file provides project rules and context for the Cursor AI assistant.

## Repository Structure
- `src/`: Main application source code.
- `tests/`: Pytest tests.
- `docs/`: Project documentation.
- `config/`: Configuration files like settings.json.

## Code Quality
- Follow PEP-8 strictly. The line length limit is 100 characters.
- All code must have Google-style docstrings (PEP-257).
- Use type hints for all function signatures and complex variables.
- Always specify `encoding='utf-8'` for file I/O operations.
- Use lazy % formatting in logging statements, not f-strings, for performance.
- All HTTP requests must include a `timeout` argument.

## Testing
- Always create or update tests for new code or bug fixes.
- Tests are located in the `tests/` directory and use the pytest framework.
- Strive for high test coverage.

## Documentation
- Keep the `docs/` directory up-to-date with any changes.
- All modules, classes, and functions must have comprehensive docstrings.
- Update `README.md` when adding or changing major features.

## Imports
- Group imports in the following order:
  1. Standard library imports
  2. Third-party library imports
  3. Local application imports (from `src/`)
- Use absolute imports for local modules (e.g., `from src.core.utils import ...`).

## UI/UX (Web Components)
- Conform to platform settings, especially dark mode.
- Aim for a minimalist and clean interface.
- Use the Pointer Events API for user interactions.
- Use absolute positioning for draggable UI elements like cards, tiles, or panels.

## Security
- Never hardcode secrets or credentials. Use environment variables or a secrets management system.
- Validate all external inputs and user-provided data.
- Sanitize all file paths and URLs to prevent path traversal and other attacks.