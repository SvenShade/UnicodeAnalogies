#!/bin/bash

python main.py --model blind --rtpath ./dataset_splits/experiment_1 --setdir Ar-N-D-N-H

python main.py --model blind --rtpath ./dataset_splits/experiment_1 --setdir Co-N-D-N-H

python main.py --model blind --rtpath ./dataset_splits/experiment_1 --setdir Di-N-D-N-H

python main.py --model blind --rtpath ./dataset_splits/experiment_1 --setdir Pr-N-D-N-H

python main.py --model blind --rtpath ./dataset_splits/experiment_1 --setdir Un-N-D-N-H


python main.py --model resnet --rtpath ./dataset_splits/experiment_1 --setdir Ar-N-D-N-H

python main.py --model resnet --rtpath ./dataset_splits/experiment_1 --setdir Co-N-D-N-H

python main.py --model resnet --rtpath ./dataset_splits/experiment_1 --setdir Di-N-D-N-H

python main.py --model resnet --rtpath ./dataset_splits/experiment_1 --setdir Pr-N-D-N-H

python main.py --model resnet --rtpath ./dataset_splits/experiment_1 --setdir Un-N-D-N-H


python main.py --model relbase --rtpath ./dataset_splits/experiment_1 --setdir Ar-N-D-N-H

python main.py --model relbase --rtpath ./dataset_splits/experiment_1 --setdir Co-N-D-N-H

python main.py --model relbase --rtpath ./dataset_splits/experiment_1 --setdir Di-N-D-N-H

python main.py --model relbase --rtpath ./dataset_splits/experiment_1 --setdir Pr-N-D-N-H

python main.py --model relbase --rtpath ./dataset_splits/experiment_1 --setdir Un-N-D-N-H


python main.py --model mrnet --rtpath ./dataset_splits/experiment_1 --setdir Ar-N-D-N-H

python main.py --model mrnet --rtpath ./dataset_splits/experiment_1 --setdir Co-N-D-N-H

python main.py --model mrnet --rtpath ./dataset_splits/experiment_1 --setdir Di-N-D-N-H

python main.py --model mrnet --rtpath ./dataset_splits/experiment_1 --setdir Pr-N-D-N-H

python main.py --model mrnet --rtpath ./dataset_splits/experiment_1 --setdir Un-N-D-N-H


jac-run main.py --model scl --rtpath ./dataset_splits/experiment_1 --setdir Ar-N-D-N-H

jac-run main.py --model scl --rtpath ./dataset_splits/experiment_1 --setdir Co-N-D-N-H

jac-run main.py --model scl --rtpath ./dataset_splits/experiment_1 --setdir Di-N-D-N-H

jac-run main.py --model scl --rtpath ./dataset_splits/experiment_1 --setdir Pr-N-D-N-H

jac-run main.py --model scl --rtpath ./dataset_splits/experiment_1 --setdir Un-N-D-N-H
