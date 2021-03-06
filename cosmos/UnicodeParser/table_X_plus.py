import os
import re
import string
import pandas as pd
from stanfordnlp.server import CoreNLPClient
from db_models import Equation, Variable, TableX
from sqlalchemy.dialects import postgresql
from fonduer.meta import Meta
from fonduer.parser.models import Sentence

from stanfordnlp.server import CoreNLPClient

db_connect_str = "postgres://postgres:password@localhost:5432/cosmos_unicode_1"

stop_words = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "eq", "eqs", "equation", "equations", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "us", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

class parseTree(object):
    """Parse tree class to represent the parse tree of each sentence"""
    def __init__(self, dep='root', pos='NULL', begin_char=-1, value = '', isUsed = False, children=None):
        self.dep = dep
        self.pos = pos
        self.begin_char = begin_char
        self.value = value
        self.isUsed = isUsed
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.print_tree(0)
    def add_child(self, node):
        assert isinstance(node, parseTree)
        self.children.append(node)
    def print_tree(self, num_of_indent):
        info = ''
        info += '\t'*num_of_indent
        info += self.dep+'--'+self.value+'--'+self.pos+'--'+str(self.begin_char)+'\n'
        for child in self.children:
            info += child.print_tree(num_of_indent+1)
        return info

def contain_alpha(word):
    """
    Check if a word contains at least one alphabet.
    :param word: Input word.
    :return: True or False
    """
    for char in word:
        if char.isalpha():
            return True
    return False

def alpha_ratio(word):
    """
    Calculate the alphabet ratio of a word.
    :param word: Input word.
    :return: Alphabet ratio of a word. 
    """
    num = 0
    for char in word:
        if char >= 'a' and char <= 'z':
            num += 1
    return float(num)/len(word)

def get_edges_by_source_id(edges, source_id):
    """
    Get the edges from the specified source.
    :param edges: Input edges.
    :param source_id: Id of the source.
    :return: List of edges
    """
    res = []
    for edge in edges:
        if edge.source == source_id:
            res.append(edge)
    return res

def subTreeConstruct(edges, root_id, dep_type, index_base, token_value, token_pos, token_begin_char):
    """
    Construct a subtree recursively
    :param edges: Edges of the subtree.
    :param root_id: Id of the root.
    :param dep_type: Dependency type.
    :param index_base: Base of the index.
    :param token_value: Text of the root.
    :param token_pos: Part of Speech tag of the root.
    :param token_begin_char: Position of the begin char of the root in the sentence.
    :return: Root node of the subtree.
    """
    node = parseTree(
                    dep=dep_type, 
                    pos=token_pos[index_base+root_id-1], 
                    begin_char=token_begin_char[index_base+root_id-1],
                    value = token_value[index_base+root_id-1],
                    )
    for edge in get_edges_by_source_id(edges, root_id):
        child =  subTreeConstruct(edges, edge.target, edge.dep, index_base, token_value, token_pos, token_begin_char)
        node.add_child(child)
    return node

def token_begin_char_calibrate(token_value, text):
    """
    Calibrate the begin char position of each token in the sentence.
    :param token_value: text of the token
    :param text: Sentence text
    :return: Calibrated positions
    """
    res = {}
    ptr = 0
 
    for key in sorted(token_value.keys()):
        token = token_value[key]
        ptr_t = 0
        while ptr_t < len(token):
            if ptr >= len(text):
                print('Calibration failed!')
                return None
            if text[ptr] == token[ptr_t]:
                if ptr_t == 0:
                    res[key] = ptr
                ptr += 1
                ptr_t += 1
            else:
                ptr += 1           
            
    assert len(token_value) == len(res)
    return res

def parseTreeConstruct(sentences, text):
    """
    Construct the parse tree.
    :param sentences: Sentence objects from the CoreNLP.
    :param text: Text of the sentence.
    :return: A list of tree roots.
    """
    trees = []
    index_base = 0
    for sent in sentences:
        #Get token information, the index starts from 0 and doesn't reset at the beginning of the next sentence
        token_value = {}
        token_pos = {}
        token_begin_char = {}

        for token in sent.token:
            index = token.tokenBeginIndex
            token_value[index] = token.originalText
            token_pos[index] = token.pos
            token_begin_char[index] = token.beginChar

        if token_begin_char_calibrate(token_value, text):
            token_begin_char = token_begin_char_calibrate(token_value, text)

        dependency_parse = sent.basicDependencies
        if len(dependency_parse.root) != 1:
            print('The number of root is '+str(len(dependency_parse.root)))
        root_id = dependency_parse.root[0]
        edges = dependency_parse.edge
        tree = subTreeConstruct(edges, root_id, 'root', index_base, token_value, token_pos, token_begin_char)
        trees.append(tree)

        index_base += len(token_value)

    return trees

def get_token_index(char_positions, text):
    """
    Get the index of a token in the text.
    :param char_positions: Position of the begin character.
    :param text: Text of the sentence.
    :return: A list of indices.
    """
    space = [' ','\n','\t']
    res = []
    index = 0

    start = False
    
    for i in range(len(text)-1):
        if text[i] not in space:
            start = True
        if text[i] in space and text[i+1] not in space and start:
            index += 1
        if i in char_positions:
            res.append(index)
    if len(text)-1 in char_positions:
        res.append(index)
    return res

def isBetween(x, a, b):
    """
    Check if x is between a and b.
    :param x: x.
    :param a: a.
    :param b: b.
    :return: True or False.
    """
    if x > min(a,b) and x < max(a,b):
        return True
    return False

def get_phrase_based_on_NN(node, text, parent_id):
    """
    Get a phrase based on the input noun node.
    :param node: Root of the tree (should be a noun).
    :param text: Text of the sentence.
    :parent_id: Id of the parent node.
    :return: Phrase extracted (a dictionary of tokens).
    """
    MODS = ['amod','nummod','compound','cc']
    AUX = ['det','case']
    tokens = {}
    if not node.pos.startswith('NN'):
        print('Should be NN, but the pos is '+node.pos)
        return tokens
    this_node_index = get_token_index([node.begin_char], text)[0]
    tokens[this_node_index] = node.value
    node.isUsed = True
    for child in node.children:
        if child.dep in MODS and child.pos != 'FW':
            child_node_index = get_token_index([child.begin_char], text)[0]
            tokens[child_node_index] = child.value
            child.isUsed = True
        if parent_id > 0 and  child.dep in AUX:
            child_node_index = get_token_index([child.begin_char], text)[0]
            if isBetween(child_node_index, parent_id, this_node_index):
                tokens[child_node_index] = child.value
                child.isUsed = True
        if child.dep == 'nmod' and child.pos.startswith('NN'):
            child_tokens = get_phrase_based_on_NN(child, text, this_node_index)
            tokens = {**tokens, **child_tokens}
        if child.dep == 'conj' and child.pos.startswith('NN'):
            child_tokens = get_phrase_based_on_NN(child, text, this_node_index)
            tokens = {**tokens, **child_tokens}
    return tokens

def cc_strip(tokens):
    """
    Strip unnecessary conjunction words.
    :param tokens: Input tokens.
    :return: Stripped tokens.
    """
    cc = ['and', 'but', 'for', 'nor', 'or', 'so', 'and', 'yet']
    sortedKey = sorted(tokens.keys())
    if tokens[sortedKey[0]].lower() in cc:
        tokens.pop(sortedKey[0])
    if tokens[sortedKey[-1]].lower() in cc:
        tokens.pop(sortedKey[-1])
    return tokens

def get_phrases(tree, text, indices_banned):
    """
    Extracted phrases from a tree.
    :param tree: Root of the tree.
    :param text: Text of the sentence.
    :param indices_banned: Indices of the tokens not to be considered.
    :return: A list of extracted sentences.
    """
    phrases = []
    if len(get_token_index([tree.begin_char],text)) != 1:
        print(tree)
        print(text)
    index = get_token_index([tree.begin_char],text)[0]
    if tree.pos.startswith('NN') and tree.isUsed == False and not index in indices_banned and contain_alpha(tree.value):
        tokens = get_phrase_based_on_NN(tree, text, -1)
        tokens = cc_strip(tokens)
        if len(tokens) > 0:
            phrases.append(tokens)
    for child in tree.children:
        child_phrases = get_phrases(child, text, indices_banned)
        phrases += child_phrases
    return phrases


def remove_symbol(phrases, indices_banned, text_banned):
    """
    Remove symbols from the extraced sentences.
    :param phrases: Input phrases.
    :param indices_banned: Indices of the tokens to be removed.
    :param text_banned: Text to be removed.
    :return: Phrases after removing.
    """
    res = []
    for phrase in phrases:
        for key in phrase.keys():
            token = phrase[key]
            if key not in indices_banned and alpha_ratio(token) > 0.5 and alpha_ratio(token)*len(token) > 3.5 and token not in text_banned and token.lower() not in stop_words:
                res.append(phrase)
                break
    return res

def build_table_X(db, corenlp):
    """
    Build the table containing the symbols and phrases for each equation.
    :param db: db connection string.
    :param corenlp: Location of the CoreNLP java file.
    """
    os.environ["CORENLP_HOME"] = corenlp

    session = Meta.init(db).Session()
    variables = session.query(Variable).order_by(Variable.equation_id)
    equations = session.query(Equation).order_by(Equation.id)
    sentences = session.query(Sentence).order_by(Sentence.id)
    with CoreNLPClient(annotators=['pos','depparse']) as client:
        vars_used_index = []
        vars_used_text = []

        for eqt in equations:
            vars_in_eqt = variables.filter(Variable.equation_id == eqt.id).order_by(Variable.sentence_id)
            if vars_in_eqt.count() == 0:
                print('No Variable found for equation ' + str(eqt.id))
            else:
                vars_used = []
                sent_used = []
                entities = []
                phrases_top = []
                phrases_bottom = []
                phrases_left = []
                phrases_right = []
                phrases_page = []
                
                for var in vars_in_eqt:
                    vars_used_index.append((eqt.document_id, var.sentence_id, var.sentence_offset))
                    PUNC_TO_STRIP = [',','.','?','(',')','{','}','[',']']
                    text = var.text
                    for punc in PUNC_TO_STRIP:
                        text = text.strip(punc)
                    vars_used_text.append((eqt.document_id, text))
                    if text not in vars_used:
                        vars_used.append(text)
                for var in vars_in_eqt:
                    sent_id = var.sentence_id
                    target_sent = sentences.filter(Sentence.id == sent_id)[0]

                    top = target_sent.top
                    bottom = target_sent.bottom
                    left = target_sent.left
                    right = target_sent.right
                    page = target_sent.page

                    indices_banned = [t[2] for t in vars_used_index if t[0] == eqt.document_id and t[1] == sent_id]
                    text_banned = [t[1] for t in vars_used_text if t[0] == eqt.document_id]
                    
                    if sent_id not in sent_used:
                        sent_used.append(sent_id)
                        sent_text = var.sentence_text
                        ann = client.annotate(sent_text)

                        sentences_ann = ann.sentence
                        trees = parseTreeConstruct(sentences_ann, sent_text)
                        for tree in trees:
                            phrases = get_phrases(tree, sent_text, [])
                            phrases = remove_symbol(phrases, indices_banned, text_banned)
                            for phrase in phrases:
                                phrase_text = ''
                                top_tmp = []
                                bottom_tmp = []
                                left_tmp = []
                                right_tmp = []
                                page_tmp = []
                                top_str = ''
                                bottom_str = ''
                                left_str = ''
                                right_str = ''
                                page_str = ''

                                for key in sorted(phrase.keys()):
                                    phrase_text += phrase[key]+' '
                                    top_tmp.append(top[key])
                                    bottom_tmp.append(bottom[key])
                                    left_tmp.append(left[key])
                                    right_tmp.append(right[key])
                                    page_tmp.append(page[key])

                                entities.append(phrase_text)                                      

                                df = pd.DataFrame({'top':top_tmp,'bottom':bottom_tmp, 'left':left_tmp, 'right':right_tmp, 'page':page_tmp})
                                maxV = df.groupby('top').max()
                                minV = df.groupby('top').min()

                                for index, row in minV.iterrows():
                                    top_str += str(index)
                                    top_str += ' ' 
                                    left_str += str(row['left'])
                                    left_str += ' '
                                    bottom_str += str(row['bottom'])
                                    bottom_str += ' '
                                    page_str += str(row['page'])
                                    page_str += ' '
                                for index, row in maxV.iterrows():
                                    right_str += str(row['right'])
                                    right_str += ' '
                                phrases_top.append(top_str)
                                phrases_left.append(left_str)
                                phrases_right.append(right_str)
                                phrases_bottom.append(bottom_str)
                                phrases_page.append(page_str)
                                    
                
                x = TableX(
                    equation_id=eqt.id,
                    symbols=vars_used,
                    phrases=entities,
                    phrases_top = phrases_top,
                    phrases_bottom = phrases_bottom,
                    phrases_left = phrases_left,
                    phrases_right = phrases_right,
                    phrases_page = phrases_page,
                )

                session.add(x)

        session.commit()


if __name__ == '__main__':
    build_table_X(db_connect_str)
