
"""
Define constants for semeval-10 task.
"""
EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids
SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'ORGANIZATION': 2, 'PERSON': 3}

OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'PERSON': 2, 'ORGANIZATION': 3, 'DATE': 4, 'NUMBER': 5, 'TITLE': 6, 'COUNTRY': 7, 'LOCATION': 8, 'CITY': 9, 'MISC': 10, 'STATE_OR_PROVINCE': 11, 'DURATION': 12, 'NATIONALITY': 13, 'CAUSE_OF_DEATH': 14, 'CRIMINAL_CHARGE': 15, 'RELIGION': 16, 'URL': 17, 'IDEOLOGY': 18}

NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6, 'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}

# POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46, 'pad': 47}

# DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'punct': 2, 'compound': 3, 'case': 4, 'nmod': 5, 'det': 6, 'nsubj': 7, 'amod': 8, 'conj': 9, 'dobj': 10, 'ROOT': 11, 'cc': 12, 'nmod:poss': 13, 'mark': 14, 'advmod': 15, 'appos': 16, 'nummod': 17, 'dep': 18, 'ccomp': 19, 'aux': 20, 'advcl': 21, 'acl:relcl': 22, 'xcomp': 23, 'cop': 24, 'acl': 25, 'auxpass': 26, 'nsubjpass': 27, 'nmod:tmod': 28, 'neg': 29, 'compound:prt': 30, 'mwe': 31, 'parataxis': 32, 'root': 33, 'nmod:npmod': 34, 'expl': 35, 'csubj': 36, 'cc:preconj': 37, 'iobj': 38, 'det:predet': 39, 'discourse': 40, 'csubjpass': 41}
POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1,'$': 33,
 "''": 22,
 ',': 48,
 '-LRB-': 13,
 '.': 25,
 ':': 47,
 'ADD': 5,
 'AFX': 9,
 'CC': 38,
 'CD': 44,
 'DT': 39,
 'EX': 11,
 'FW': 37,
 'HYPH': 10,
 'IN': 24,
 'JJ': 42,
 'JJR': 8,
 'JJS': 43,
 'LS': 29,
 'MD': 7,
 'NFP': 32,
 'NN': 20,
 'NNP': 15,
 'NNPS': 14,
 'NNS': 26,
 'PDT': 17,
 'POS': 41,
 'PRP': 4,
 'PRP$': 34,
 'RB': 23,
 'RBR': 16,
 'RBS': 2,
 'RP': 40,
 'SYM': 31,
 'TO': 36,
 'UH': 30,
 'VB': 27,
 'VBD': 45,
 'VBG': 21,
 'VBN': 6,
 'VBP': 46,
 'VBZ': 12,
 'WDT': 28,
 'WP': 19,
 'WP$': 3,
 'WRB': 18,
 '``': 35}
# DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'punct': 2, 'compound': 3, 'case': 4, 'nmod': 5, 'det': 6, 'nsubj': 7, 'amod': 8, 'conj': 9, 'dobj': 10, 'ROOT': 11, 'cc': 12, 'nmod:poss': 13, 'mark': 14, 'advmod': 15, 'appos': 16, 'nummod': 17, 'dep': 18, 'ccomp': 19, 'aux': 20, 'advcl': 21, 'acl:relcl': 22, 'xcomp': 23, 'cop': 24, 'acl': 25, 'auxpass': 26, 'nsubjpass': 27, 'nmod:tmod': 28, 'neg': 29, 'compound:prt': 30, 'mwe': 31, 'parataxis': 32, 'root': 33, 'nmod:npmod': 34, 'expl': 35, 'csubj': 36, 'cc:preconj': 37, 'iobj': 38, 'det:predet': 39, 'discourse': 40, 'csubjpass': 41}
DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'acl': 23,
 'acl:relcl': 11,
 'advcl': 14,
 'advmod': 6,
 'amod': 35,
 'appos': 27,
 'aux': 3,
 'aux:pass': 25,
 'case': 22,
 'cc': 38,
 'cc:preconj': 34,
 'ccomp': 36,
 'compound': 7,
 'compound:prt': 33,
 'conj': 15,
 'cop': 32,
 'csubj': 31,
 'csubj:pass': 30,
 'det': 42,
 'det:predet': 28,
 'discourse': 44,
 'expl': 21,
 'fixed': 43,
 'flat': 26,
 'iobj': 19,
 'list': 4,
 'mark': 37,
 'nmod': 10,
 'nmod:npmod': 20,
 'nmod:poss': 2,
 'nmod:tmod': 41,
 'nsubj': 17,
 'nsubj:pass': 13,
 'nummod': 24,
 'obj': 5,
 'obl': 9,
 'obl:npmod': 40,
 'obl:tmod': 18,
 'parataxis': 12,
 'punct': 8,
 'root': 16,
 'vocative': 29,
 'xcomp': 39 }
NEGATIVE_LABEL = 'Other'

# LABEL_TO_ID = {'Other': 0, 'Entity-Destination': 1, 'Cause-Effect': 2, 'Member-Collection': 3, 'Entity-Origin': 4, 'Message-Topic': 5, 'Component-Whole': 6, 'Instrument-Agency': 7, 'Product-Producer': 8, 'Content-Container': 9, 'Entity-Destination-rev': 10, 'Cause-Effect-rev': 11, 'Member-Collection-rev': 12, 'Entity-Origin-rev': 13, 'Message-Topic-rev': 14, 'Component-Whole-rev': 15, 'Instrument-Agency-rev': 16, 'Product-Producer-rev': 17, 'Content-Container-rev': 18} 
LABEL_TO_ID = {'Other': 0, 'Entity-Destination': 1, 
'Cause-Effect': 2, 'Member-Collection': 3, 'Entity-Origin': 4,
 'Message-Topic': 5, 'Component-Whole': 6, 'Instrument-Agency': 7, 'Product-Producer': 8, 'Content-Container': 9}

INFINITY_NUMBER = 1e12

















