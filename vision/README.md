### Usage

`python get_interactions.py -o <output_path_for_tensor> -p <if passed uses phi in calculation term>  -c <if passed uses cifar test set> -n <num samples> -r <img refrence value>`

### List of Experiments:

| Dataset | Phi | Reference | Complete |
|-----|-----|-----|----|
| MNIST | True | 0 |  |
| MNIST | False | 0 | |
| MNIST | True | noise | |
| MNIST | False | noise | |
| CIFAR | True | 0 | |
| CIFAR | False | 0 | |
| CIFAR | True | noise | |
| CIFAR | False | noise | |
