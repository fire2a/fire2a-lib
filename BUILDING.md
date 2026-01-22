# Building

### Pre-requisites
```
pip install -r requirements.build.txt
pip install -e .
```

### Distribution
#### Project Overview
- `pyproject.toml` specified
- [src-layout](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#src-layout) organized files
- its [API documentation](https://fire2a.github.io/fire2a-lib/) web is automated with [pdoc](https://pdoc.dev)
- dynamic versioning with `setuptools_csm`
- unit tests with [pytest](https://pytest.org)
- built & published on a gitlab-ci [pipeline](https://github.com/fire2a/fire2a-lib/actions/workflows/publish-pypi.yml) running on the [latest-qgis](https://registry.hub.docker.com/r/qgis/qgis) docker container running
- [published](https://pypi.org/project/fire2a-lib/) on [pypi](https://pypi.org)
  
#### Manual Steps
```bash
# Ensure working tree is clean
git status

# Clean (-n is dry-run, remove to delete)
git clean -dfX -n
git clean -dfX      # X removes only .gitignored files
git clean -dfx -n
git clean -dfx      # x removes all untracked (danger!)

# Calculate the next tag
git tag --sort=-version:refname -n | head
python -m setuptools_scm

# Create tag LOCALLY first (don't push yet)
git tag -a v0.3.13 -m 'preventive breaking changes with pandas v3'

# NOW check version is clean
python -m setuptools_scm

# Test build locally: creates `dist` with .whl & tar.gz files
python -m build

# Push tag (triggers GitHub Actions workflow for PyPI publishing)
git push origin v0.3.13

# [if failure] Undo delete tag locally & upstream
git tag --delete v0.3.13 && git push --delete origin v0.3.13
```

### Documentation
Build the full static webpage locally, using [pdoc](https://pdoc.dev) to generate the code api documentation.

Check the [how-to](https://pdoc.dev/docs/pdoc.html#how-can-i) for essential options.

#### Manual Steps
If directory exists remove, then build
```bash
rm -rf doc/*
touch doc/.gitkeep
pdoc --output-directory doc --math --show-source --logo https://www.fire2a.com/static/img/logo_1_.png --favicon https://www.fire2a.com/static/img/logo_1_.png fire2a
```

Then check the generated webpage
```bash
firefox doc/index.html
```
