#!/bin/bash

python main.py  --model blind   --rtpath ./dataset_splits/experiment_4 --setdir challenge --kfolds 1

python main.py  --model resnet  --rtpath ./dataset_splits/experiment_4 --setdir challenge --kfolds 1

python main.py  --model relbase --rtpath ./dataset_splits/experiment_4 --setdir challenge --kfolds 1

python main.py  --model mrnet   --rtpath ./dataset_splits/experiment_4 --setdir challenge --kfolds 1 --epochs 15

jac-run main.py --model scl     --rtpath ./dataset_splits/experiment_4 --setdir challenge --kfolds 1
