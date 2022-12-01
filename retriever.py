import os
import sys
import heapq
import numpy as np
import time
from tokenizer import tokenize
import nltk.probability as probability
from utils import load_json, load_jsonl
from indexer import get_term_posting, index_dir


doc_index_path = index_dir + "\\doc_index.json"
termID_index_path = index_dir + "\\termID_index.jsonl"
partial_index_path = index_dir + "\\partial_indexes\\partial_index"
term_filepos_index_path = index_dir + "\\term_filepos_index.json"
merged_index_path = index_dir + "\\merged_index.jsonl"
corpus_stats_path = index_dir + "\\corpus_stats.json"


# preprocess query and return the normalized vector representation and the partial term index
def preprocess_query(query, index_file, term_filepos_index, termID_index, max_terms=15, stopword_threshold=0.25):
    terms = termID_index["term"].keys()
    query_tokens = tokenize(query, stopword=True, terms=terms)
    query_tokens_stopwords = tokenize(query, stopword=False, terms=terms)   # query tokens with stopwords

    if len(query_tokens) > max_terms:  # Only take the first max_terms tokens of query
        query_tokens = query_tokens[:max_terms]
    # Check if query_tokens is within stopword_threshold compared to query_token_stopwords
    elif len(query_tokens_stopwords)*stopword_threshold > len(query_tokens):
        query_tokens = query_tokens_stopwords

    fdist = probability.FreqDist(termID_index["term"][t] for t in query_tokens)

    term_index = dict()     # partial index with only terms in query
    # print("GETTING TERMPOSTING")
    # t = time.time()
    for term_id in sorted(fdist.keys()):    # sorted by term_id reading from file is more efficient
        term_index[term_id] = get_term_posting(term_id, term_filepos_index, index_file)
    # print(time.time() - t)
    # print("FNISHED GETTING TERMPOSTING")

    vector_rep = dict()
    length = 1  # length of document vector representation (start with 1 to avoid divided by 0)

    for term_id in fdist.keys():
        vector_rep[term_id] = (1 + np.log(fdist[term_id])) * term_index[term_id]["idf"]
        length += vector_rep[term_id] ** 2
    length = np.sqrt(length)
    for term_id in fdist.keys():  # Normalize
        vector_rep[term_id] = vector_rep[term_id] / length
    return vector_rep, term_index


def evaluate_score(query_vector_rep, term_index, doc_id):
    a_cos = 1
    a_tf = 0.01
    score = 0
    for term_id in term_index.keys():
        if doc_id in term_index[term_id]["document"].keys():
            score += a_cos * term_index[term_id]["normalized"][doc_id]*query_vector_rep[term_id]    # Cosine similarity
            score += a_tf * term_index[term_id]["document"][doc_id]*term_index[term_id]["idf"]      # tfidf
    return score


def search(query, index_file, term_filepos_index, termID_index, doc_index, corpus_stats, top_k=20, terms_matched_threshold=0.7):
    query_vector_rep, term_index = preprocess_query(query, index_file, term_filepos_index, termID_index)

    result = dict()
    # Scoring documents from tiers
    for i in range(len(corpus_stats["tier"])+1):
        documents = list()
        if len(result) >= top_k:
            break
        for term_id in term_index.keys():
            docs = term_index[term_id]["tier"][i]
            documents.extend(docs)
        # Make an freqDist for documents indicate how many of query tokens in the document
        documents_fdist = probability.FreqDist(d for d in documents)

        # Calculating score for documents
        skipped_document = set()
        for d in documents_fdist.keys():
            # Only consider documents matched over threshold
            if d in result.keys():
                continue
            if documents_fdist[d] < len(query_vector_rep)*terms_matched_threshold:
                skipped_document.add(d)
                continue    # Pass document match less term than threshold
            result[d] = evaluate_score(query_vector_rep, term_index, d)
        # If len(result) is not top_k then keep going through skipped documents
        for d in skipped_document:
            if len(result) >= top_k:
                break
            result[d] = evaluate_score(query_vector_rep, term_index, d)

    # Getting top_k result using heap
    heap_result = list((-value, key) for key, value in result.items())
    heapq.heapify(heap_result)
    for i in range(top_k):
        if len(heap_result) == 0:
            break
        res = heapq.heappop(heap_result)
        print(doc_index[int(res[1])]["url"], -res[0])


def input_loop(index_file, term_filepos_index, termID_index, doc_index, corpus_stats, top_k=20):
    while True:
        query = input("SEARCH QUERY: ")
        if len(query.strip()) == 0:
            break
        # print("QUERY:", query)
        t = time.time()
        search(query, index_file, term_filepos_index, termID_index, doc_index, corpus_stats, top_k=top_k)
        print("FINISED IN", time.time() - t, "\n")


def search_pipeline(top_k=20):
    print("LOADING AUXILIARIES DATA STRUCTURE...")
    term_filepos_index = load_json(term_filepos_index_path)
    print("term_filepos_index size:", sys.getsizeof(term_filepos_index), "bytes")
    termID_index = load_jsonl(termID_index_path)
    print("termID_index size:", sys.getsizeof(termID_index), "bytes")
    doc_index = load_json(doc_index_path)
    print("doc_index size:", sys.getsizeof(doc_index), "bytes")
    corpus_stats = load_json(corpus_stats_path)
    index_file = open(merged_index_path, "r")
    # get_term_posting(0, term_filepos_index, index_file)
    print("FINISH LOADING")

    try:
        input_loop(index_file, term_filepos_index, termID_index, doc_index, corpus_stats, top_k=top_k)
    except Exception as e:
        raise e
    finally:
        index_file.close()


if __name__ == "__main__":
    search_pipeline()
