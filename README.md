<span style="color: orange;"><strong><em><a href="https://fire2a.github.io/docs/docs/qgis-toolbox/README.html" style="color: orange;">
Friendly graphical user interface click here
</a></em></strong></span>

![PyPI workflow](https://github.com/fire2a/fire2a-lib/actions/workflows/publish-pypi.yml/badge.svg)
![auto pdoc workflow](https://github.com/fire2a/fire2a-lib/actions/workflows/auto-docs.yml/badge.svg)
![PyPI Version](https://img.shields.io/pypi/v/fire2a-lib.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/fire2a-lib.svg)
![License](https://img.shields.io/github/license/fire2a/fire2a-lib.svg)
![Downloads](https://img.shields.io/pypi/dm/fire2a-lib.svg)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)

# fire2a-lib python package
Fire Advanced Analyitics research group scriptable knowledge base
- Novel Algorithms related to landscape risk metrics, spatial decision optimization models, etc.
- (Cell2) Fire (W) Simulator integration algorithms to facilitating placing firebreaks, measuring forest fire impacts, etc.
- (Q)GIS algorithms to common tasks related to rasters, polygons, spatial clustering, etc.

## Quickstart
### Simplest: Use within QGIS
1. Install QGIS
2. Open QGIS Python Console
3. Type (to install)::
```python
!pip install fire2a-lib
```
4. Visit [fire2a-lib documentation](https://fire2a.github.io/fire2a-lib), example:
```python
from fire2a.downstream_protection_value import digraph_from_messages
digraph = digraph_from_messages('messages01.csv')
```
### Command line usage
1. Install QGIS
2. Use python from the terminal
- Windows: Open OsGeo4W Shell app
- MacOs: Open terminal app and type `alias pythonqgis=/Applications/QGIS.app/Contents/MacOS/bin/python`, use that python in all subsequent commands
- Linux: Do a system site packages aware python virtual environment `python3 -m venv --system-site-packages qgis_venv`
3. Install
```bash
python -m pip install fire2a-lib
```
4. Visit [fire2a-lib documentation](https://fire2a.github.io/fire2a-lib), example:
```bash
python -m fire2a.cell2fire -vvv --base-raster ../fuels.asc --authid EPSG:25831 --scar-sample Grids/Grids2/F
orestGrid03.csv --scar-poly propagation_scars.shp --burn-prob burn_probability.tif
```
### Scripting
1. Check [standalone scripting](https://github.com/fire2a/fire-analytics-qgis-processing-toolbox-plugin/blob/main/script_samples/standalone.py) for more info on initializing a headless QGIS environment
2. Boring [template examples](https://github.com/fire2a/fire2a-lib/tree/main/usage_samples)
- Windows users check this VSCode [integration](https://fire2a.github.io/docs/docs/qgis-cookbook/README.html#making-a-python-environment-launcher-for-developers)
- macOS users consider adding the alias to your default terminal session by appending to `~/.zsh`
3. Interactive debug:
```
breakpoint()
from IPython import embed
embed()
```

### Interactive
IPython, qtconsole or jupyter compatible
```bash
# Interactive explore from IPython
In [1]: from fire2a.<press-tab-to-continue>

# Copy a whole module except for the last main and if __name__ == __main__ part
In [1]: %paste
# Then execute tha main method step by step

```

## Render documentation
```bash
pip install pdoc
pip install --editable .
pdoc --math --show-source fire2a
```
### Build the full static webpage
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

# [Contributing](./CODING.md)
# [Building](./BUILDING.md)


# Code of Conduct

Everyone interacting in the project's codebases, issue trackers,
chat rooms, and fora is expected to follow the
[PSF Code of Conduct](https://www.python.org/psf/conduct).
