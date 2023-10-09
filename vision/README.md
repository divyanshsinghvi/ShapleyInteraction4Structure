### Usage

`python get_interactions.py -o <output_path_for_tensor> -p <if passed uses phi in calculation term>  -c <if passed uses cifar test set> -m <if passed uses mnist test set> -n <num samples> -r <img refrence value>`

**Currently only reference value is 0 => so we set `-r 0`**

### List of Experiments:

| # | Dataset | Phi | Reference | Num Samples | Complete |
|-|-----|-----|-----|---|----|
|1| MNIST | True | 0 |  | | 
|2| MNIST | False | 0 | | |
|3| CIFAR | True | 0 | | | 
|4| CIFAR | False | 0 | | | 