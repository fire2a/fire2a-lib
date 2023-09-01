![manual workflow](https://github.com/fire2a/fire2a-lib/actions/workflows/manual.yml/badge.svg)
![auto workflow](https://github.com/fire2a/fire2a-lib/actions/workflows/auto.yml/badge.svg)
<a href=https://github.com/psf/black>![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)</a>

Welcome to Fire2a research group algorithms and tools python package.

Novel Algorithms to calculate metrics, clustering, placing firebreaks, measuring forest fire impacts, etc.

Tools related to GIS, graphs, optimization, etc.

## Documentation

__User__: https://fire2a.github.io/fire2a-lib

All our releases: https://fire2a.github.io/docs

About us: https://www.fire2a.com

__Developers__: [template package tutorial](development_tutorial.md) or https://github.com/fdobad/template-python-package

## Quickstart
Installing [QGIS](https://qgis.org), cover most of our [requirements.txt](https://raw.githubusercontent.com/fire2a/fire2a-lib/main/requirements.txt)
```bash
(qgis_python_venv) $ pip install git+https://github.com/fire2a/fire2a-lib.git
(qgis_python_venv) $ pip install -r requirements.txt
```
Any language-server-protocol enabled environment is recommended:
```bash
(qgis_python_venv) $ ipython
In [1]: from fire2a.<press-tab-to-continue>
```
### append to requirements.txt
Choose latest, branch, tag or commit
```
# latest main
fire2a-lib @ git+https://github.com/fire2a/fire2a-lib.git
# latest branch
fire2a-lib @ git+https://github.com/fire2a/fire2a-lib.git@branch
# tag
fire2a-lib @ git+https://github.com/fire2a/fire2a-lib.git@v1.2.3
# commit
fire2a-lib @ git+https://github.com/fire2a/fire2a-lib.git@e855bdb96202db42dc9013ea5c5cf934cee3f8d1
```
This is a developing repo, __anchor your code to a commit to disable any incoming (possibly breaking) changes.__

## Development Setup
__Fork it before cloning to contribute!__
```bash
git clone git@github.com:fire2a/fire2a-lib.git
cd fire2a-lib
git checkout -b my_branch
cp hook/pre-commit .git/hooks/.
chmod +x .git/hooks/pre-commit
cp hook/pre-push .git/hooks/.
chmod +x .git/hooks/pre-push
pip install -r requirements.dev.txt
pip install --editable .
pdoc --html --http : --config latex_math=True fire2a  
```
GDAL is not listed on requirements!  
Install QGIS, make a venv with system-site-packages flag

### Live view a single installed package
```bash
pip install --requirement requirements.doc.txt
pip install --editable .
pdoc --html --http : --config latex_math=True <fire2 package name>
```
Packages are directories under `src` with at least a `__init__.py` file inside

### Build the full static webpage
if directory exists remove, then build
```bash
if [ -d doc/fire2a-lib ]; then
    rm -r doc/fire2a-lib
fi
pdoc --html --force --output-dir doc --filter=src,tests --config latex_math=True .
```

### Uninstall
```bash
pip uninstall fire2a-lib
```

# Code of Conduct

Everyone interacting in the project's codebases, issue trackers,
chat rooms, and fora is expected to follow the
[PSF Code of Conduct](https://www.python.org/psf/conduct).
