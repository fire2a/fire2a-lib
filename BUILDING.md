# Building
## Project Overview
- `pyproject.toml` specified
- [src-layout](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#src-layout) organized files
- its [API documentation](https://fire2a.github.io/fire2a-lib/) web is automated with [pdoc](https://pdoc.dev)
- dynamic versioning with `setuptools_csm`
- unit tests with [pytest](https://pytest.org)
- built & published on a gitlab-ci [pipeline](https://github.com/fire2a/fire2a-lib/actions/workflows/publish-pypi.yml) running on the [latest-qgis](https://registry.hub.docker.com/r/qgis/qgis) docker container running
- [published](https://pypi.org/project/fire2a-lib/) on [pypi](https://pypi.org)
  
## Manual Steps
```bash
# clean (-n is dry-run, remove to delete)
git clean -dfX -n
git clean -dfX

# tags check
git tag

# tag create locally & upload
git tag -a v0.3.4 -m 'multi objective knapsack mvp + cell2fire statistics mean fix without non-fire scenarios' && git push origin v0.3.4

# delete tag locally & upstream
git tag --delete v0.3.4 && git push --delete origin v0.3.4

# view calculated version to check is not dirty
python -m setuptools_scm

# build : creates `dist` with .whl & tar.gz files
python -m build
```

## Build the full static webpage locally
```bash
if directory exists remove, then build
```bash
rm -rf doc/*
touch doc/.gitkeep
pdoc --output-directory doc --math --show-source --logo https://www.fire2a.com/static/img/logo_1_.png --favicon https://www.fire2a.com/static/img/logo_1_.png fire2a fire2template
```
Then check the generated webpage
```bash
firefox doc/index.html
```
