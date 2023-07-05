#!/bin/bash

python main.py --model blind --rtpath ./dataset_splits/experiment_2 --setdir GLOB-A-N-D-N-H

python main.py --model blind --rtpath ./dataset_splits/experiment_2 --setdir LOC-A-N-D-N-H

python main.py --model blind --rtpath ./dataset_splits/experiment_2 --setdir OBJ-A-N-D-N-H


python main.py --model resnet --rtpath ./dataset_splits/experiment_2 --setdir GLOB-A-N-D-N-H

python main.py --model resnet --rtpath ./dataset_splits/experiment_2 --setdir LOC-A-N-D-N-H

python main.py --model resnet --rtpath ./dataset_splits/experiment_2 --setdir OBJ-A-N-D-N-H


python main.py --model relbase --rtpath ./dataset_splits/experiment_2 --setdir GLOB-A-N-D-N-H

python main.py --model relbase --rtpath ./dataset_splits/experiment_2 --setdir LOC-A-N-D-N-H

python main.py --model relbase --rtpath ./dataset_splits/experiment_2 --setdir OBJ-A-N-D-N-H


python main.py --model mrnet --rtpath ./dataset_splits/experiment_2 --setdir GLOB-A-N-D-N-H

python main.py --model mrnet --rtpath ./dataset_splits/experiment_2 --setdir LOC-A-N-D-N-H

python main.py --model mrnet --rtpath ./dataset_splits/experiment_2 --setdir OBJ-A-N-D-N-H


jac-run main.py --model scl --rtpath ./dataset_splits/experiment_2 --setdir GLOB-A-N-D-N-H

jac-run main.py --model scl --rtpath ./dataset_splits/experiment_2 --setdir LOC-A-N-D-N-H

jac-run main.py --model scl --rtpath ./dataset_splits/experiment_2 --setdir OBJ-A-N-D-N-H
