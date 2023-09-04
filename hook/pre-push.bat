@echo off

rem Build the documentation
if exist doc\fire2a-lib (
    rmdir /s /q doc\fire2a-lib
)
call python-qgis-venv.bat
pdoc --html --force --output-dir doc --filter=src,tests --config latex_math=True .

rem Add the modified files to the commit
git add doc\fire2a-lib\*
