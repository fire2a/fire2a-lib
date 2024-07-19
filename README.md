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
- Docker users check [qgis](https://hub.docker.com/r/qgis/qgis) container
- Linux developers could skip QGIS by `apt install python3-gdal gdal-bin`
2. Use python from the terminal
- Windows: Open OsGeo4W Shell app, `python` is available
- MacOS: Open terminal app, use this `/Applications/QGIS.app/Contents/MacOS/bin/python` (see creating an alias below)
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
### Scripting tips
- Check [standalone scripting](https://github.com/fire2a/fire-analytics-qgis-processing-toolbox-plugin/blob/main/script_samples/standalone.py) for more info on initializing a headless QGIS environment
- Usage [examples](https://github.com/fire2a/fire2a-lib/tree/main/usage_samples)
- Microsoft users check this VSCode [integration](https://fire2a.github.io/docs/docs/qgis-cookbook/README.html#making-a-python-environment-launcher-for-developers)
- macOS users add a permanent alias, on the terminal app
```zsh
echo "alias pythonq='/Applications/QGIS.app/Contents/MacOS/bin/python'" >> ~/.zshrc
```
#### Interactive 
- debbuging:
```
breakpoint()
from IPython import embed
embed()
```
- sessions, with: IPython, qtconsole, jupyter-lab, or IPyConsole (QGIS plugin) compatible
```bash
# Interactive explore from IPython
In [1]: from fire2a.<press-tab>

# Select and Copy a whole module from line 1 up -but not included- to 'def main def main(argv=None):' line 
In [2]: %paste

# Choose your args 
In [3]: args = arg_parser.parse_args(['-vvv', '--base-raster', ...
# Skip:
    if argv is sys.argv:
        argv = sys.argv[1:]
    args = arg_parser(argv)

# Ready
```

# [Contributing](./CODING.md)
# [Building](./BUILDING.md)


# Code of Conduct

Everyone interacting in the project's codebases, issue trackers,
chat rooms, and fora is expected to follow the
[PSF Code of Conduct](https://www.python.org/psf/conduct).
