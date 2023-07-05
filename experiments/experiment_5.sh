#!/bin/bash

python main.py  --model blind   --rtpath ./dataset_splits/experiment_5 --setdir Co-V-N-N-H --epochs 10

python main.py  --model resnet  --rtpath ./dataset_splits/experiment_5 --setdir Co-V-N-N-H --epochs 10

python main.py  --model relbase --rtpath ./dataset_splits/experiment_5 --setdir Co-V-N-N-H --epochs 10

python main.py  --model mrnet   --rtpath ./dataset_splits/experiment_5 --setdir Co-V-N-N-H --epochs 10

jac-run main.py --model scl     --rtpath ./dataset_splits/experiment_5 --setdir Co-V-N-N-H --epochs 10
