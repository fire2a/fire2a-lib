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
# clean (-n is dry-run, remove to delete)
git clean -dfX -n
git clean -dfX
git clean -dfx -n
git clean -dfx

# tags check
git tag

# Create tags on the main branch!
# tag create locally & upload
git tag -a v0.3.11 -m 'wind flip' && git push origin v0.3.11

# delete tag locally & upstream
git tag --delete v0.3.11 && git push --delete origin v0.3.11

# view calculated version to check is not dirty
python -m setuptools_scm

# build : creates `dist` with .whl & tar.gz files
python -m build
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
