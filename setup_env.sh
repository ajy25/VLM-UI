#!/bin/bash


cd "$(dirname "$0")"

python3 vlmui/models/CheXagent/download.py &
python3 vlmui/models/xraygpt/download.py &

wait

echo "Setup scripts have finished running."