import spacy
import networkx as nx
from itertools import combinations
from transformers import GPT2TokenizerFast
from transformers import BertTokenizerFast


def get_spacy_pipeline(model = 'gpt2'):

    nlp = spacy.load("en_core_web_sm")

    if model == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    elif model == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def GPT2Tokenize(text):
        tokens = tokenizer.tokenize(text)
        return spacy.tokens.Doc(nlp.vocab, words=tokens)

    def Bert2Tokenize(text):
        tokens = tokenizer.tokenize(text)
        return spacy.tokens.Doc(nlp.vocab, words=tokens)


    if model == 'gpt2':
        nlp.tokenizer = GPT2Tokenize
    elif model == 'bert':
        nlp.tokenizer = Bert2Tokenize
    
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

def get_syntactic_distance(doc:spacy.tokens.Doc, index=False, return_graph=False) -> dict:

    edges = []
    for token in doc:
        edges.extend([(token.i, child.i) for child in token.children])
    
    G = nx.from_edgelist(edges)
    combs = list(combinations(list(range(len(doc))), 2))
    syntactic_mapping = {}

    for c in combs:
        if c[0] in G.nodes and c[1] in G.nodes:
            if index:
                key = (c[0], c[1])
            else:
                key = (doc[c[0]], doc[c[1]])
            try:
                syntactic_mapping[key] = nx.shortest_path_length(G, source=c[0], target=c[1])    
            except nx.NetworkXNoPath:
                syntactic_mapping[key] = float('-inf')


    if return_graph:
        return syntactic_mapping, G
    return syntactic_mapping
