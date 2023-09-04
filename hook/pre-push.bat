
cd doc
IF EXIST fire2a-lib DEL /F /S /Q fire2a-lib
call python-qgis-env.bat
cd ..
pdoc --html --force --output-dir doc --filter=src,tests --config latex_math=True .
# Add the modified files to the commit
git add doc/fire2a-lib/*