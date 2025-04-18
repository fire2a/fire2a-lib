# Setup QGIS python launcher
To
1. Select it as VSCode project's default interpreter
2. Setup any cmd terminal or make a launchable icon
3. Setup as VSCode project's default terminal

You need to get the QGIS install location:
1. Launch `OSGeo4W shell` application
2. Run the command:
  - 2.A. `echo %OSGEO4W_ROOT%`
  - 2.B. Or `for %i in ("%CD%") do @echo %~fsi`

You'll get QGIS install location similar to this: `C:\PROGRA~1\QGIS33~1.3`

## 1. Select it as VSCode project's default interpreter 
Beware that after selecting it, warnings will appear about dev-tools not being installed in this environment, pip install them (see requirements.code.txt).

A. Using the `Command Palette` (ctrl+shift+p):  
`> Select Interpreter`  
`<QGIS install location>/bin/python-qgis.bat`  
-example: `C:/PROGRA~1/QGIS33~1.2/bin/python-qgis.bat`

B. [alternative] Project configuration file, write or append to `<MyProject>/.vscode/settings.json`
```json
{
  "python.defaultInterpreterPath": "C:\\PROGRA~1\\QGIS33~1.3\\bin\\python-qgis.bat",
}
```
C. For unwritable bundled python, check the more complicated: 
```json
{
  "python.defaultInterpreterPath": ".\\qgis-launchers\\python-qgis-path.bat",
}
```

## 2. Setup any cmd terminal
To make a location independent launcher to enable any terminal to run python-qgis, we'll edit copy and edit `python-qgis.bat` to be run anywhere
```batch
# 1. open OSGeo4W Shell application

# 2. annotate the install location
>cd
C:\Program Files\QGIS 3.38.2

# 3. copy 
copy bin\python-qgis.bat <your\launcher\path\>activate-python-qgis.bat

# 4. edit activate-python-qgis.bat
# 4.1. replace "%~dp0" for the install location on line 2:
call "C:\Program Files\QGIS 3.38.2\bin\o4w_env.bat"

# 4.2. replace the last line "python %*" with "cmd.exe /k %*" if you want to double click the icon to open a terminal
# Else delete the line and use it `call activate-python-qgis.bat` in scripts (losing the icon launcher usefulness)
```

## 3. Setup as VSCode project's default terminal
A. Project configuration file, write or append to `<MyProject>/.vscode/settings.json`
```json
{
  "terminal.integrated.defaultProfile.windows": "QGIS Python CMD",
  "terminal.integrated.profiles.windows": {
    "QGIS Python CMD": {
      "path": "cmd.exe",
      "args": [
        "/k",
        "call .\\qgis-launchers\\activate-python-qgis.bat"
      ]
    }
  }
}
```
Here the location of the launcher `call qgis-launchers/activate-python-qgis.bat` is relative to the [fire2a-lib](https://github.com/fire2a) repo.  
Adjust accordingly to where you created it in step 2.

B. For unwritable bundled python, check the more complicated:
```json
    "QGIS Python +Path CMD": {
      "path": "cmd.exe",
      "args": [
        "/k",
        "call .\\qgis-launchers\\activate-python-qgis-path.bat"
      ]
    }
  }
```