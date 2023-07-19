**WORK IN PROGRESS, moving out of notebook once get past initial stage**

How to use:

1. Recreate environment from `environment.yaml` (I use conda, but nltk, spacy, and cython are main dependencies so if you just copy those versions should be ok)
2. Follow instructions for creating document in `data_processing/process_wiki.ipynb`
3. The document may have some empty lines which will throw an error in the MWE tagger, for this we can run `./preprocess.sh document` (I made this quickly for wiki dataset so may still have errors for other datasets). Be careful as this replaces strings in document.
4. Run `./sst.sh document` on the final document.
5. The output will be `document.pred.sst` and `document.pred.tags`, I am currently working on parsing these for the next stage.



## Note:

There are still some differences between python 2 and python 3 outputs.

Presumably, to see these samples we can compare the `PY2` and `PY3` pred tags files if interested.

## TODO: 

Complete BPE
