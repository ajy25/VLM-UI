#!/bin/bash


cd "$(dirname "$0")"

python3 vlmui/models/chexagent_model/download.py &
python3 vlmui/models/xraygpt_model/download.py &
python3 vlmui/models/biomedgpt_model/download.py &

wait

echo "Setup scripts have finished running."