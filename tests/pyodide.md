# Testing with pyodide

## Prerequisites

1. This must be run on unix
2. Make sure node is installed

       sudo apt install node

## Install tests requirements and pytest-pyodide

    pip install -r tests/requirements.txt

## Install pyodide

Update the version and run the following snippet:

    wget https://github.com/pyodide/pyodide/releases/download/0.26.2/pyodide-core-0.26.2.tar.bz2
    tar xjf pyodide-core-0.26.2.tar.bz2
    mv pyodide pyodide-dist/

## Run tests in pyodide

    pytest --run-in-pyodide --dist-dir=pyodide-dist
