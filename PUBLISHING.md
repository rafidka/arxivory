# Publishing to PyPI

This document explains how to publish the `arxivory` package to PyPI.

## Prerequisites

1. **PyPI Account**: Create an account at [pypi.org](https://pypi.org/account/register/)
2. **GitHub Repository**: Push this code to a GitHub repository
3. **PyPI Trusted Publishing**: Configure trusted publishing for secure, token-free deployment

## Setting up PyPI Trusted Publishing

1. Go to [PyPI Trusted Publishers](https://pypi.org/manage/account/publishing/)
2. Click "Add a new pending publisher"
3. Fill in the details:
   - **PyPI project name**: `arxivory`
   - **Owner**: Your GitHub username (e.g., `rafidka`)
   - **Repository name**: `arxivory`
   - **Workflow filename**: `publish.yml`
   - **Environment name**: (leave empty)

## Publishing Process

### Automatic Publishing (Recommended)

The package will automatically be published to PyPI when you:

1. **Create a GitHub Release**:

   ```bash
   # Tag and push a new version
   git tag v0.3.0
   git push origin v0.3.0

   # Or create a release through GitHub web interface
   ```

2. **The workflow will**:
   - Run linting and type checks
   - Test CLI functionality
   - Build the package
   - Publish to PyPI automatically

### Manual Publishing

You can also trigger publishing manually:

1. Go to your GitHub repository
2. Navigate to **Actions** → **Publish to PyPI**
3. Click **Run workflow** → **Run workflow**

## Version Management

Update the version in `pyproject.toml` before creating releases:

```toml
[project]
name = "arxivory"
version = "0.4.0"  # Update this
```

## Local Testing

Test the package locally before publishing:

```bash
# Build and check
uv build
uv run twine check dist/*

# Install locally and test
pip install dist/arxivory-*.whl
arxivory --help
```

## CI/CD Workflows

- **`ci.yml`**: Runs on every push/PR for linting and testing
- **`publish.yml`**: Runs on releases for automated PyPI publishing

Both workflows use:

- Python 3.12
- uv for dependency management
- ruff for linting and formatting
- pyright for type checking
