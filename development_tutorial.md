![manual workflow](https://github.com/fdobad/template-python-package/actions/workflows/manual.yml/badge.svg)
![auto workflow](https://github.com/fdobad/template-python-package/actions/workflows/auto.yml/badge.svg)
<a href=https://github.com/psf/black>![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)</a>

__version__ = 'v0.0.1-0-gecfa54c-dirty'

# Template Overview

* A [source layout][src-layout] python project/distribution with 2 packages/modules
* [pyproject.toml][pyproject_config] configuration file
* Auto-documentation using pdoc3, publishing pages in `doc/my-repo-name` to [`https://user.github.io/my-repo-name`](https://fdobad.github.io/template-python-package)
* A simple precommit hooks to auto-update versions in python files

# Goal

An easy-to-document&version python distribution/project, by self publishing code docstrings and being pip installable.

* Easy to install:
```bash
pip install git+https://github.com/user/repo.git
```

* To version:
```bash
pip install git+https://github.com/user/repo.git@SUFFIX
# SUFFIX CAN BE:
tip of branch:      @branch
specific commit:    @commit_id  (40 digits long SHA-hash not 7)
tag (not dirty):    @tag
...
pkg: @[tag|branch]#egg=mypackage
faster: pip install https://github.com/user/repository/archive/branch.[zip|wheel]
```
Or as a line in requirements.txt:
```
my-project-name @ git+https://github.com/user/repo.git@SUFFIX
```

* To develop, install with editable flag:
```bash
git clone git@git...
cd repo
# git checkout -b my_new_feature
# pip install -r requirements.dev.txt
# git clean -dfX -n
pip install -e .
pdoc --http localhost:8080 --config latex_math=True mypkg
```

# Daily usage 

Activate you python venv, check if/where is installed: 
```bash
$ pip list | grep my-project-name
my-project-name           0.0.0       /home/fdo/source/template-python-package
```
* If a filepath to the local repo is shown and --editable was used: the project can be directly edited & the documentation served live  
* Else from a commit, be sure it is pointing to the correct one on requirements.txt
* Or uninstall and focus on testing

Development tips:
* Beware of python sessions not reloading the entire packages when `import mypkg`, these should be restarted, not reset.
* Version control tracked files get automatically added to the release, so be tidy, aware of `.gitignore` & beware `git add .`
* Do your feature branch to contribute! `git checkout -b my_new_feature`
* Use docstrings and typed methods to build a great documentation! Even latex (to html or pdf) is available. [more info](https://pdoc3.github.io/pdoc/doc/pdoc/#what-objects-are-documented). For a method, describe input `parameters & returns`, additionaly `exceptions, classes, variables, module header`, `docstring examples` are not tested at the time, use `pytest` instead
    
* All new `.py` files should have at least `version` & `author` in the header, sample:
```python
#!python3
""" 
this text is the header module docstring
"""
__author__ = "Software Authors Name"
__copyright__ = "Copyright (C) 1 May 1886 Author Name"
__license__ = "https://anticapitalist.software"
__version__ = "v0.0.1"
```

# Setup pattern

1. Fork, rename, clone
2. Enable github pages for the repo ?
3. Enable the version-auto-updating, web-page-creating hooks:
```bash
cp hooks/* .git/hooks/.
chmod +x .git/hooks/pre-commit .git/hooks/pre-push
```
4. Install requirements.dev.txt
5. Change something, rebuilding the docs (`pdoc`) or run tests (`pytest`)
6. Push (to feature branch)

# Development

## live
```bash
pip install --editable .
pdoc --http localhost:8080 --config latex_math=True mypkg
pdoc --http localhost:8081 --config latex_math=True otherpkg
```
## [testing with pytest][pytest]
```
pip uninstall my-project-name
pytest
```
## building
```bash
# just in case upgrade build tool
python -m pip install --upgrade build
# creates dist folder with .whl & tar.gz
python -m build
# recreates docs
pdoc --html --force --output-dir doc .
```
## troubleshoot
Build files are not automatically cleaned!
```bash
# check first -n is --dry-run
git clean -dfX -n
# once sure, delete them
git clean -dfX
```

# github actions
There're two actions (in .github/workflow) to make the web page:

* manual: needs updating the doc before pushing  
* auto: builds the doc online, needs updated requirements.txt or pyproject.toml:dependencies  

The headers needs some configuration (uncommenting) to work, a good practice is to enable them `on branch doc`, to not update needlessly (only 2000 free mins a month)

# pdoc
```bash
pdoc --html --force --output-dir doc .
pdoc --html --http : --force --output-dir doc --config latex_math=True .
pdoc --html --http localhost:8080 --force --output-dir doc mypkg
```
https://github.com/pdoc3/pdoc/blob/master/pdoc/templates/config.mako

# git tag
```bash
# list tags
git tag

# tag current branch-commit
git tag -a v0.0.2 -m "message"

# delete 
## local
git tag --delete v0.0.2
## remote
git push --delete origin v0.0.2

# push 
## a tag
git push origin v0.0.2
## all tags
git push origin --tags
```

# References
* [learn about pytest][pytest]
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
[pytest]: https://docs.pytest.org/en/latest/getting-started.html
