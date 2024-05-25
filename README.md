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

1. Installing [QGIS](https://qgis.org), cover most of our [requirements.txt](https://raw.githubusercontent.com/fire2a/fire2a-lib/main/requirements.txt).

2. Activate QGIS python environment:  
__Windows__ users run `python-qgis-cmd.bat` in a `cmd terminal`. [more info](https://fire2a.github.io/docs/docs/qgis/README.html#making-an-environment-launcher).  
__Linux__ users: just launch QGIS from a venv activated terminal.

3. Install
```bash
(qgis_python_venv) $ pip install git+https://github.com/fire2a/fire2a-lib.git
(qgis_python_venv) $ pip install -r requirements.txt
```

4. Use
Any language-server-protocol enabled environment is recommended:
```bash
(qgis_python_venv) $ ipython
In [1]: from fire2a.<press-tab-to-continue>
```
## On your project: append to requirements.txt
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
# choose your hook [if any, very optional]
cp hooks/* .git/hooks/.
chmod u+x .git/hooks/pre-commit .git/hooks/pre-push
#
## activate QGIS python environment (check next paragraph)
#
pip install -r requirements.dev.txt
pip install --editable .
pdoc --html --http : --config latex_math=True fire2a  
```
Watch out! GDAL is not listed on requirements! The easiest way is to install QGIS, then:  
__Linux__ users should make a venv with the `--system-site-packages` flag  
__Windows + VSCode__ users: this repo includes `.vscode\settings.json` that automates enabling the environment by  calling `python-qgis-cmd.bat`, so they should:
- (all users will be affected) Make their QGIS python env [writable](https://fire2a.github.io/docs/docs/qgis/README.html#make-it-writable)
- Open this repo folder as new window
- (Ctrl+Shift+P) 'Python: Select Interpreter' > ... Select interpreter path: C:\Program Files\QGIS 3.32.2\apps\Python\python.exe
- (Ctrl+Shift+P) 'Python: Create Terminal' enables the environment, __now you can pip install!__
- __Bewate: git push__ only works from the terminal! Not the buttons!
- Also, selecting code and pressing Shift+Enter executes it on the python terminal.  
A nice trick is exiting python and then opening ipython  
Then select, send with 'Shift+Enter', then Ctrl+` to switch between code and terminal windows

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

### Build and versioning
```bash
# clean (-n is dry-run, remove to delete)
git clean -dfX -n

# tag
git tag -a v1.2.3 -m 'message'
git push origin v1.2.3

# view calculated version to check is not dirty
python -m setuptools_scm

# build : creates `dist` with .whl & tar.gz files
python -m build
```

### Uninstall
```bash
pip uninstall fire2a-lib
```
### Troubleshoot
1. Can't commit or push, some deleted file is messing git up: Temporarily delete .git/hooks/pre-commit and/or .git/hooks/pre-push; restore them when wanting to publish another webpage

# Coding style
[Here](./coding_style.md)  
TL;DR: Use black formatter, pytests; write standard docstrings and avoid needless complexity.

# Code of Conduct

Everyone interacting in the project's codebases, issue trackers,
chat rooms, and fora is expected to follow the
[PSF Code of Conduct](https://www.python.org/psf/conduct).
