# Contributing to the project
## Overview
1. Fork
2. Clone
3. Branch named as the new feature or bug you are working on
4. Python virtual environment setup
5. Code with style
6. Commit often, create a legible history
7. Assert your work with tests
8. Prettify your docummentation
9. Pull Request

## Descriptive
### 1. Fork
Fire2a org. members can skip this
### 2. Clone
```bash
git clone git@github.com:fire2a/fire2a-lib.git
cd fire2a-lib
# forgot to fork and have local changes? fork and change the remote url
git remote set-url origin git@github.com:<YOUR-USER-HERE>/fire2a-lib.git
```
### 3. Create a branch 
```bash
git switch --create <feature-issue-branch>
```
### 4. Setup (QGIS aware) python virtual environment
```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# python -m pip install --upgrade -r requirements.txt
pip install --editable .
```
### 5. Code with style 
```bash
# use these linters, formatters, fixers:
pip install -r requirements.code.txt
# check configurations at [tool.*] in pyproject.toml
# start with the template
cp src/fire2template/template.py src/fire2a/<new-feature-module>.py
```
### 6. Commit often
- Create a legible history
- Explain your code/changes with multiple commits (atomic, descriptive)
### 7. Assert your work with tests
```bash
pytest tests/test_<new-feat>.py
```
### 8. Prettify your documentation 
#### live @ localhost:8080
```bash
pip install pdoc
pdoc --math fire2template fire2a
```

# Some coding principles

1. The best code is the easiest to read, not the most efficient

1. Avoid reinventing the wheel: Research alternatives before coding from scratch

1. Each function/method should:
    - Do __one__ thing 
    - Return __one__ type of object
    - Avoid void methods, at least return True or 0 for success

1. DRY: Don't repeat yourself, encapsulate/extract methods whenever possible
    - move hard-coded strings to configuration files
    - prefer functions over classes

1. Naming is key
   - Use consistent and descriptive names that your future you would understand reading two months from now
   - Avoid abbreviations, search & replace them to something descriptive before sharing the code
   - (Python-style) Variable & methods names all lowercase_with_underscores
   - (Python-style) Classes names in CamelCase

1. Documenting:
   - At first, develop using in line comments (#)
   - Afterwards replace them with docstrings and renaming variables
   - Follow the template formatting docstring convention, make sure it looks ok on the live server
   - The most important is the initial sentence that describes the method, module, class, etc. (Args, Returns, Raises sections can be AI generated)
   - Type annotations for functions, classes and notable variables

1. Unit testing
   - Avoid generating random test instances (can be non-deterministic when using without a random number seed) 
   - At first, develop directly using the `if __name__ == '__main__':` section
   - Then migrate to its own test/test_stuff.py file
