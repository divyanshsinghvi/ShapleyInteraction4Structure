import spacy
import networkx as nx
from itertools import combinations
from transformers import GPT2TokenizerFast


def get_spacy_pipeline():

    nlp = spacy.load("en_core_web_sm")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def GPT2Tokenize(text):
        tokens = tokenizer.tokenize(text)
        return spacy.tokens.Doc(nlp.vocab, words=tokens)

    nlp.tokenizer = GPT2Tokenize

    return nlp

def parse_sentences_mwe(dataset):
    lines = []
    for d in dataset:
        doc = nlp(d)
        if doc:
            for sent in doc.sents:
                for word in sent:
                    lines.append(f"{word}\t{word.pos_}\n")
                lines.append("\n")
    return lines

def parse_dependencies(pipeline, text:str) -> dict:
    """
    Use spaCy dependency parser on the text.

    If you read the streusle files in a pandas dataframe you can just run `df.apply(lambda x: parse_dependencies(x['text']), axis=1)
    """
    doc = pipeline(text)
    d = {}
    for token in doc:
        d.update({token.text: token.dep_})
    
    return d

def get_syntactic_distance(doc:spacy.tokens.Doc) -> dict:

    edges = []
    for token in doc:
        edges.extend([(token.i, child.i) for child in token.children])
    
    G = nx.from_edgelist(edges)
    combs = list(combinations(list(range(len(doc))), 2))
    syntactic_mapping = {}

    for c in combs:
        syntactic_mapping[(doc[c[0]], doc[c[1]])] = nx.shortest_path_length(G, source=c[0], target=c[1])

    return syntactic_mapping
