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
3. To install, type :
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

- 1.A. [Simple](https://qgis.org/download/)
- 1.B. Docker users check [qgis container](https://hub.docker.com/r/qgis/qgis)
- __Optional but worth it for Windows:__ Make QGIS's bundled python folder writable by your user
   - Three alternatives:
   - A. Open File Explorer, go to `C:\Program Files\QGIS 3.38.3\apps`, right-click `Python312` folder [Show more options]\Properties\Security\Edit\Allow Full control\OK
   - B. Open cmd administrator terminal, type:
     ```
     icacls "C:\Program Files\QGIS 3.38.3\apps\Python312" /grant %username%:F /T
     ```

     
2. Locate python, prepare environment

- 2.A. Windows
   - Open `OsGeo4W Shell` application, 
   - Run `bin\python-qgis.bat` once to configure all environment variables in the current session.
   - Also you could: `bin\python-qgis.bat your_script.py`
   - Recommended persistent [integration](https://github.com/fire2a/fire2a-lib/blob/main/qgis-launchers/README.md)
- 2.B. macOS 
   - Open terminal application, 
   - Use this python: `/Applications/QGIS.app/Contents/MacOS/bin/python` (see creating an alias below)
   - TODO: are all environment variables right?
- 2.C. Linux
   ```
   python3 -m venv --system-site-packages qgis_venv
   source qgis_venv/bin/activate
   ```
3. Install
```bash
python -m pip install fire2a-lib
# also recommended for development: ipython qtconsole jupyterlab
```
4. Visit [fire2a-lib documentation](https://fire2a.github.io/fire2a-lib), example for getting the burn probability from a simulator results directory:
```bash
python -m fire2a.cell2fire -vvv --base-raster ../fuels.asc --authid EPSG:25831 --scar-sample Grids/Grids2/ForestGrid03.csv --scar-poly propagation_scars.shp --burn-prob burn_probability.tif
```
## Scripting/Developing tips
- Check [standalone scripting](https://github.com/fire2a/fire-analytics-qgis-processing-toolbox-plugin/blob/main/script_samples/standalone.py) for more info on initializing a headless QGIS environment
- Usage [examples](https://github.com/fire2a/fire2a-lib/tree/main/usage_samples)
- macOS users add a permanent alias, on the terminal app
   ```zsh
   echo "alias pythonq='/Applications/QGIS.app/Contents/MacOS/bin/python'" >> ~/.zshrc
   ```
### Interactive 
1. Interactive sessions in IPython, qtconsole, jupyter-lab, or IPyConsole (QGIS plugin) compatible
```bash
# Press tab to auto-complete of available modules from IPython
In [1]: from fire2a.<press-tab>
```
2. Stop here interactive (colored terminal able):
```python
from IPython.terminal.embed import InteractiveShellEmbed
InteractiveShellEmbed()()
```
3. Stop here interactive (no colors):
```python
from IPython import embed
embed()
```
c. QGIS debugging:
```python
from qgis.PyQt.QtCore import pyqtRemoveInputHook
pyqtRemoveInputHook()
import pdb
pdb.set_trace()
from IPython.terminal.embed import InteractiveShellEmbed
InteractiveShellEmbed()()
```
3. Copy and paste the code interactive:
```bash
# Select and Copy a whole module from line 1 up -but not included- to 'def main def main(argv=None):' line 
In [2]: %paste

# Choose your args 
In [3]: args = arg_parser.parse_args(['-vvv', '--base-raster', ...

# Skip (reading args from sys.argv o main)
    if argv is sys.argv:
        argv = sys.argv[1:]
    args = arg_parser(argv)

# Ready to run the main interactively (args object ready)
```

# [Contributing](./CODING.md)
# [Building](./BUILDING.md)


# Code of Conduct

Everyone interacting in the project's codebases, issue trackers,
chat rooms, and fora is expected to follow the
[PSF Code of Conduct](https://www.python.org/psf/conduct).
