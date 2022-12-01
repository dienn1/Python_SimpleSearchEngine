import nltk.tokenize
import nltk.stem as stem
from nltk.util import ngrams
from nltk.corpus import stopwords
import re

stopwords = set(stopwords.words('english'))
stemmer = stem.PorterStemmer()


def word_tokenize(text):
    rule = r"\s|,|\.|\?|!|\/|\[|\]|\{|\}|\(|\)|:|;|—|–|-|-|_|@|\*|#|\$|&|<|>|\"|'"
    tokens_list = list()
    for word in re.split(rule, text):
        if len(word) > 1:
            tokens_list.append(word)
    return tokens_list


# Tokenize/stem text, ignore numeric str, ignore token not in terms if terms not None
def tokenize_stem(text, ngram=(2, 3), terms=None):
    tokens = word_tokenize(text)
    stemmed_tokens = list()
    for t in tokens:
        tmp = t.lower()
        if not tmp.isnumeric() and tmp.isalnum():
            tmp = stemmer.stem(tmp)
            if terms is None or tmp in terms:
                stemmed_tokens.append(tmp)
    # Generate ngram
    generate_ngrams(stemmed_tokens, ngram, terms)
    return stemmed_tokens


# Tokenize/stem text, ignore stopwords, ignore numeric str, ignore token not in terms if terms not None
def tokenize_stem_stopword(text, ngram=(2, 3), terms=None):
    tokens = word_tokenize(text)
    stemmed_tokens = list()
    for t in tokens:
        tmp = t.lower()
        if not tmp.isnumeric() and tmp.isalnum() and tmp not in stopwords:
            tmp = stemmer.stem(tmp)
            if terms is None or tmp in terms:
                stemmed_tokens.append(tmp)
    # Generate ngram
    generate_ngrams(stemmed_tokens, ngram, terms)
    return stemmed_tokens


# Generate ngram in range_
def generate_ngrams(tokens, range_, terms=None):
    ngram_tokens = list()
    for i in range_:
        ngram_tokens.extend(" ".join(ngram) for ngram in ngrams(tokens, i))
    for t in ngram_tokens:
        if terms is None or t in terms:
            tokens.append(t)
    return tokens


def tokenize(text, ngram=(2, 3), stopword=False, terms=None):
    tokens = tokenize_stem_stopword(text, ngram, terms) if stopword else tokenize_stem(text, ngram, terms)
    return tokens
