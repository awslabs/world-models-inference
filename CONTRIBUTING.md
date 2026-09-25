# Contributing to World Model Inference

Thank you for your interest in contributing! This document provides guidelines and information for contributors.

## Reporting Bugs

If you find a bug, please open an issue with:

- A clear, descriptive title
- Steps to reproduce the problem
- Expected behavior vs. actual behavior
- Your environment details (OS, AWS CLI version, instance type, region)
- Any relevant logs or screenshots

## Suggesting Features

Feature requests are welcome. Please open an issue describing:

- The problem you are trying to solve
- Your proposed solution
- Any alternatives you have considered

## Submitting Changes

1. **Fork** the repository and create your branch from `main`.
2. **Branch naming**: Use descriptive names like `feat/your-feature` or `fix/issue-description`.
3. **Write tests** for any new functionality.
4. **Ensure all tests pass** before submitting.
5. **Open a Pull Request** with a clear description of the changes and why they are needed.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/<your-fork>/world-model-accelerator.git
cd world-model-accelerator

# Frontend setup
cd frontend
npm install
npm run dev

# Backend / inference testing
cd ..
pip install -e ".[dev]"
pytest tests/ -v
```

## Code Style

- **Python**: Follow PEP 8. Use type hints where practical.
- **TypeScript/React**: Follow the existing patterns in the codebase. Use functional components and hooks.
- Keep commits focused and write clear commit messages.

## Security Issues

If you discover a security vulnerability, please do NOT open a public issue. Instead, email the maintainers directly so the issue can be addressed before public disclosure.

## Contributor License Agreement

By submitting a pull request, you agree that your contributions are licensed under the Apache License 2.0, the same license that covers this project. See the [LICENSE](LICENSE) file for details.
