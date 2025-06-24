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
    else:
        raise Exception(f"{model} not implemented")

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
    
    if len(edges)== 0:
        return {}
    G = nx.from_edgelist(edges)
    combs = list(combinations(list(range(len(doc))), 2))
    syntactic_mapping = {}

    nx.set_edge_attributes(G, values = 1, name = 'weight')
    #print(G.edges(data = True))
    fw_dict = nx.johnson(G, weight='weight')
    #fw_dict = nx.floyd_warshall(G, weight=None)
    for node1, dist_map in fw_dict.items():
        for node2, dist in dist_map.items():
            if index:
                key = (node1, node2)
            else:
                key = (doc[node1], doc[node2])

            syntactic_mapping[key] = len(dist)-1   

    if return_graph:
        return syntactic_mapping, G

    return syntactic_mapping

def map_syntactic_distance(df):
    new_map = {}
    mp = df["token_map_dict"]
    dist = df["syntactic_distance_idx"]
    for t0, t1 in dist:
        new_pair = []
        if t0 in mp:
            a = mp[t0]
        if t1 in mp:
            b = mp[t1]

        for c in mp.get(t0, []):
            for d in mp.get(t1, []):
                new_map[(c, d)] = dist[(t0, t1)]
        #new_map[tuple(map(tuple, new_pair))] = dist[(t0, t1)]

    return new_map
