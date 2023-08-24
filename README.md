
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: Black

.. image:: https://github.com/pypa/setuptools/workflows/tests/badge.svg
   :target: https://github.com/pypa/setuptools/actions?query=workflow%3A%22tests%22
   :alt: tests

.. image:: https://img.shields.io/readthedocs/setuptools/latest.svg
    :target: https://setuptools.pypa.io

# Template Overview

* A [source layout][src-layout] python project/distribution with 2 packages/modules
* [pyproject.toml][pyproject_config] configuration file
* Auto-documentation using pdoc3, publishing to github pages in `doc` & `https://repo.github.io/doc`
* A simple precommit hook to auto-update versions in python files

# Usage 

In the current python-venv, is the project/distribution currently installed pointing to github[commit|latest|branch] or local folder?
```bash
pip list | grep my-project-name
```
* If it is local, the module can be edited & the documentation served live
* Else from a commit, make sure it is pointing to the correct one on requirements.txt
* Or uninstall and focus on testing ?

Version control tracked files get installed on the build!

# Setup

1. Fork, rename, clone
2. Enable github pages for the repo
3. Copy the hook `hook/pre-commit` to `.git/hook/.` (you can delete the hook directory now)
4. Install requirements.dev.txt
5. Try rebuilding the docs and editing a file in `src`
6. Push

# Development

## live
```bash
pip install --editable .
pdoc --http localhost:8080 --config latex_math=True my-project-name
```
## testing
```
pip uninstall project-name
pdoc --http : --config latex_math=True .
pytest
```
## building
```bash
# just in case upgrade build tool
python -m pip install --upgrade build
# creates dist folder with .whl & tar.gz
python -m build
# 
pdoc --html --force --output-dir doc .
```
## pushing
```bash
# build docs 
pdoc --html --force --output-dir doc --config latex_math=True .
# push
```
# github actions
There're two actions (in .github/workflow):

* manual: needs updating the doc before pushing  
* auto: builds the doc online, needs updated requirements.txt or pyproject.toml:dependencies  

The headers needs some configuration (uncommenting) to work, a good practice is to enable them on branch docs, to not update needlessly

# pdoc
```bash
pdoc --html --force --output-dir doc .
pdoc --html --http : --force --output-dir doc --config latex_math=True .
pdoc --html --http localhost:8080 --force --output-dir doc mypkg
```
https://github.com/pdoc3/pdoc/blob/master/pdoc/templates/config.mako

# References
* [official userguide pyproject config at pypa][pyproject_config]  
* [src project layout][src-layout]  
* [auto python documentation][auto-document]  
* [auto publish documentation][auto-publish-docs]  
* [add command line scripts][cli-scripts]  

# Code of Conduct

Everyone interacting in the project's codebases, issue trackers,
chat rooms, and fora is expected to follow the
[PSF Code of Conduct](https://www.python.org/psf/conduct/)_.

[pyproject_config]: https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html
[src-layout]: https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#src-layout
[cli-scripts]: https://setuptools.pypa.io/en/latest/userguide/entry_point.html
[auto-document]: https://pdoc3.github.io/pdoc
[auto-publish-docs]: https://github.com/mitmproxy/pdoc/blob/main/.github/workflows/docs.yml
