![manual workflow](https://github.com/fire2a/fire2a-lib/actions/workflows/manual.yml/badge.svg)
![auto workflow](https://github.com/fire2a/fire2a-lib/actions/workflows/auto.yml/badge.svg)
<a href=https://github.com/psf/black>![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)</a>

__version__ = 'e76e8e5-dirty'

Welcome to Fire2a research group algorithms and tools python package.

Novel Algorithms to calculate metrics, clustering, placing firebreaks, measuring forest fire impacts, etc.

Tools related to GIS, graphs, optimization, etc.

## Documentation

User : https://fire2a.github.io/fire2a-lib

[Developer](development_tutorial.md) or https://github.com/fdobad/template-python-package

## Quickstart
### new python virtual environment
```bash
python3 -m venv fire
source fire/bin/activate
(fire) $ pip install git+https://github.com/fire2a/fire2a-lib.git
(fire) $ python
>>> import fire2a
```
### append to requirements.txt, choose
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
This is a developing repo, disable updates (with possible breaking changes) by choosing a tag or commit.
## Development Setup
```bash
git clone git@github.com:fire2a/fire2a-lib.git
cd fire2a-lib
cp hook/pre-commit .git/hooks/.
chmod +x .git/hooks/pre-commit
pip install --editable .
```

## Usage 
To create the static webpage before upload it use 
```bash
pdoc --html --force --output-dir doc --config latex_math=True .
```
To live prevew the pakage documentation make user you have installed the pakage and the requirements
```bash
pip install --editable .
pip install --requirement requirements.doc.txt
pdoc --html --http : fire2a
```

If you wan to uninstall the package use
```bash
pip uninstall fire2a-lib
```

# Code of Conduct

Everyone interacting in the project's codebases, issue trackers,
chat rooms, and fora is expected to follow the
[PSF Code of Conduct](https://www.python.org/psf/conduct/).
