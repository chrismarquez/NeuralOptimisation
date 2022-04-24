#/bin/bash

mkdir samples
mkdir trained
mkdir trained/metadata
python3 -m venv venv
source /venv/bin/activate
pip install -r requirements.txt