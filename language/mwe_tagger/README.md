# MWE Tagger Pipeline

This repository provides a pipeline to tag Multi-Word Expressions (MWEs) using the [`pysupersensetagger`](https://github.com/nschneid/pysupersensetagger).

Below are the detailed steps to set up the environment and run the full workflow.

---


## 📦 Environment Setup

1️⃣ **Create the Conda environment**

1. conda env create --name mwe_tagger --file conda.yaml
2. conda activate mwe_tagger

## END to END Script to run the full mwe_tagger pipeline 
3. ./run_sr_pipeline.sh <some_file_name> <model_name>

Supported: 
<model_name> : gpt2 , bert

** Detailed STEP BY STEP**

Go through the run_sr_pipeline.sh for understanding the flow.

1. Recreate environment from conda.yaml
2. ./process_hf_dataset.py <input>
3. The document may have some empty lines which will throw an error in the MWE tagger, for this we can run `./preprocess.sh document` (I made this quickly for wiki dataset so may still have errors for other datasets). Be careful as this replaces strings in document.
4. Run `./sst.sh document` on the final document.
5. The output will be `document.pred.sst` and `document.pred.tags`.
5. Then run src/tags2mwe document.pred.tags > document.pred.mwe
6. Then run `make_csv.py -f document.pred.mwe -m <model>.csv` and it will output a csv with delimeter `;` to the specified path. The column description are as follows here: https://github.com/nschneid/pysupersensetagger/tree/master.~



## Note:

There are still some differences between python 2 and python 3 outputs.
Presumably, to see these samples we can compare the `PY2` and `PY3` pred tags files if interested.
