#!/bin/bash
set -e

# Check if the number of arguments is between 1 and 3
if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
	echo "Usage: $0 <module_name> [<docstring_start> <docstring_end>]"
	echo "Example: $0 fire2a.raster.gdal_calc_sum '<!-- BEGIN_ARGPARSE_DOCSTRING -->' '<!-- END_ARGPARSE_DOCSTRING -->'"
	exit 1
fi

MODULE_NAME=$1
DOCSTRING_START=${2:-'<!-- BEGIN_ARGPARSE_DOCSTRING -->'}
DOCSTRING_END=${3:-'<!-- END_ARGPARSE_DOCSTRING -->'}
echo "Updating docstring for module: $MODULE_NAME"
echo "Docstring start: $DOCSTRING_START"
echo "Docstring end: $DOCSTRING_END"

# Generate the help message
HELP_MESSAGE=$(python -m $MODULE_NAME -h)

# Get the file path of the module
FILE_PATH=$(python -c "
import importlib
module = importlib.import_module('$MODULE_NAME')
print(module.__file__)
")

# paranoid
cp "$FILE_PATH" /tmp

# Check if the start and end tags are present in the file
if ! grep -q "$DOCSTRING_START" "$FILE_PATH"; then
	echo "Error: Start tag '$DOCSTRING_START' not found in file '$FILE_PATH'"
	exit 1
fi

if ! grep -q "$DOCSTRING_END" "$FILE_PATH"; then
	echo "Error: End tag '$DOCSTRING_END' not found in file '$FILE_PATH'"
	exit 1
fi

# Read the file and update the docstring
awk -v start="$DOCSTRING_START" -v end="$DOCSTRING_END" -v help="$HELP_MESSAGE" '
BEGIN { in_docstring = 0 }
{
    if ($0 ~ start) {
        in_docstring = 1
        print
        print help
    } else if ($0 ~ end) {
        in_docstring = 0
        print
    } else if (!in_docstring) {
        print
    }
}
' "$FILE_PATH" >"${FILE_PATH}.tmp"

# Replace the original file with the updated file
mv "${FILE_PATH}.tmp" "$FILE_PATH"
