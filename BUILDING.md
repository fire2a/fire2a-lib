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

# tags check
git tag
# tag create locally
git tag -a v1.2.3 -m 'message'
# tag upload
git push origin v1.2.3

# oops: revert tag
git tag --delete v1.2.3 && git push --delete origin v1.2.3

# view calculated version to check is not dirty
python -m setuptools_scm

# build : creates `dist` with .whl & tar.gz files
python -m build
```

