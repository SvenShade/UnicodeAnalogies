#!/bin/bash

python main.py --model blind --rtpath ./dataset_splits/experiment_3 --setdir A-N-D-N-H

python main.py --model blind --rtpath ./dataset_splits/experiment_3 --setdir A-N-D-S-H

python main.py --model blind --rtpath ./dataset_splits/experiment_3 --setdir A-E-D-S-H

python main.py --model blind --rtpath ./dataset_splits/experiment_3 --setdir A-P-D-S-H


python main.py --model resnet --rtpath ./dataset_splits/experiment_3 --setdir A-N-D-N-H

python main.py --model resnet --rtpath ./dataset_splits/experiment_3 --setdir A-N-D-S-H

python main.py --model resnet --rtpath ./dataset_splits/experiment_3 --setdir A-E-D-S-H

python main.py --model resnet --rtpath ./dataset_splits/experiment_3 --setdir A-P-D-S-H


python main.py --model relbase --rtpath ./dataset_splits/experiment_3 --setdir A-N-D-N-H

python main.py --model relbase --rtpath ./dataset_splits/experiment_3 --setdir A-N-D-S-H

python main.py --model relbase --rtpath ./dataset_splits/experiment_3 --setdir A-E-D-S-H

python main.py --model relbase --rtpath ./dataset_splits/experiment_3 --setdir A-P-D-S-H


python main.py --model mrnet --rtpath ./dataset_splits/experiment_3 --setdir A-N-D-N-H

python main.py --model mrnet --rtpath ./dataset_splits/experiment_3 --setdir A-N-D-S-H

python main.py --model mrnet --rtpath ./dataset_splits/experiment_3 --setdir A-E-D-S-H

python main.py --model mrnet --rtpath ./dataset_splits/experiment_3 --setdir A-P-D-S-H


jac-run main.py --model scl --rtpath ./dataset_splits/experiment_3 --setdir A-N-D-N-H

jac-run main.py --model scl --rtpath ./dataset_splits/experiment_3 --setdir A-N-D-S-H

jac-run main.py --model scl --rtpath ./dataset_splits/experiment_3 --setdir A-E-D-S-H

jac-run main.py --model scl --rtpath ./dataset_splits/experiment_3 --setdir A-P-D-S-H
