# CODING STYLE

## Python tools 
python-language-server python-lsp, with:
- formatter: black 
	--line-length=120
	# fmt: skip/on/off
- linter: pylint 
	--max-line-length=120
- type checker: python-mypy
	install stubs
- code actions: python-rope

pip:
* see requirements.dev.txt
* see pyproject.toml

vscode extensions:
	ms-python.black-formatter
	ms-python.vscode-pylance
	ms-python.python 

## Guiding principles

* avoid complexity
  - each function should
    - do one thing 
    - return one type of object
    - try avoiding not returning anything
  - extract 
    - methods where possible
    - hard-coded strings where where possible
  - prefer functions over classes

* descriptive naming is key
  - avoid abbreviations, search & replace them to something descriptive before sharing the code
  - variable & methods names all lowercase_with_underscores
  - classes names in CamelCase

* documenting:
  - at first use # line comments, afterwards replace them with docstrings and renaming variables
  - docstring convention
    - a sentence that describes what the function does
    - Args, Returns, Raises sections
  - type annotations for functions, classes and notable variables

- botar argparse de python
- mejora de argparse para especificar cada raster y no una carpeta
- desde asc hasta tif
- listado archivos output
- futuro: ver si pasamos a tif con hartas bands o gpkg
