import os
import numpy as np
import json
import nltk.probability as probability
from bs4 import BeautifulSoup
import time
from tokenizer import tokenize
from utils import load_json, load_jsonl, dump_json, dump_jsonl, initialize_directory, dump_jsonl_sorted, read_chunk
from filter import FilterDuplicate

index_dir = "index"

doc_index_path = index_dir + "\\doc_index.json"
termID_index_path = index_dir + "\\termID_index.jsonl"
partial_index_path = index_dir + "\\partial_indexes\\partial_index"
term_filepos_index_path = index_dir + "\\term_filepos_index.json"
merged_index_path = index_dir + "\\merged_index.jsonl"
corpus_stats_path = index_dir + "\\corpus_stats.json"


def build_doc_index(doc_dir):
    doc_index = list()  # {dir, url}
    filter_duplicate = FilterDuplicate()
    print("Building DocumentID Index...")
    for dir_name, subdir_list, file_list in os.walk(doc_dir, topdown=False):
        print(dir_name)
        for f in file_list:
            with open(dir_name + "\\" + f, "r") as file:
                data = json.load(file)
            url_defrag = filter_duplicate.add_url(data["url"])  # check for url duplication after defrag
            if url_defrag is not None:
                doc_index.append({"dir": dir_name + "\\" + f, "url": url_defrag})
    dump_json(doc_index, doc_index_path)
    print()
    return doc_index


def build_termID_index(doc_index):
    tokens_count = 0
    termID_index = {"term": dict(), "id": list()}   # "term": access id with term, "id": access term with id
    print("Building TermID Index...\n")
    for d in doc_index:
        with open(d["dir"], "r") as file:
            data = json.load(file)
        soup = BeautifulSoup(data["content"], "lxml")
        stemmed_tokens = tokenize(soup.text)
        for t in stemmed_tokens:
            if t not in termID_index["term"].keys():
                termID_index["id"].append(t)
                termID_index["term"][t] = tokens_count
                tokens_count += 1
    dump_jsonl(termID_index, termID_index_path)
    return termID_index


# Build partial indexes and update termID_index if neccessary
def build_partial_index(doc_index, termID_index=None):
    partition = 10
    partition_len = 1 + int(len(doc_index)/partition)
    doc_count = 0   # count docs to partition_len
    partition_count = 0   # count number of partial indexes made
    bold_multiplier = 10
    index = dict()  # Structure: {tokenID: "document":{document_id:tf}, "idf": int}
    filter_duplicate = FilterDuplicate()

    non_duplication_count = 0

    if termID_index is None:
        tokens_count = 0
        termID_index = {"term": dict(), "id": list()}  # "term": access id with term, "id": access term with id
    else:
        tokens_count = len(termID_index)

    print("Building partial indexes...\n")
    for i in range(len(doc_index)):
        with open(doc_index[i]["dir"], "r") as file:
            data = json.load(file)
        soup = BeautifulSoup(data["content"], "lxml")
        stemmed_tokens = tokenize(soup.get_text(separator=" "))
        not_dup = filter_duplicate.add_tokens(stemmed_tokens)  # check for duplicate content
        if not not_dup:
            continue
        non_duplication_count += 1
        bold = soup.find_all("b")
        bold = " ".join(b.get_text(separator=" ") for b in bold)
        stemmed_bold = tokenize(bold)

        # Update termID_index
        for t in stemmed_tokens:
            if t not in termID_index["term"].keys():
                termID_index["id"].append(t)
                termID_index["term"][t] = tokens_count
                tokens_count += 1

        # Indexing
        fdist = probability.FreqDist(termID_index["term"][t] for t in stemmed_tokens)
        for b in stemmed_bold:  # Give bolded words more frequency
            if b in termID_index["term"].keys():
                fdist[termID_index["term"][b]] += bold_multiplier
        length = 0  # length of document vector representation
        for token in fdist.keys():
            if token not in index:
                index[token] = {"document": dict(), "normalized": dict()}
            index[token]["document"][i] = 1 + np.log(fdist[token])
            length += index[token]["document"][i] ** 2
        length = np.sqrt(length)
        for token in fdist.keys():  # Normalize
            index[token]["normalized"][i] = index[token]["document"][i] / length

        # Check to store partial index
        doc_count += 1
        if doc_count >= partition_len:
            dump_jsonl_sorted(index, partial_index_path + str(partition_count) + ".jsonl")
            partition_count += 1
            doc_count = 0
            index = dict()
    # dump the rest
    dump_jsonl_sorted(index, partial_index_path + str(partition_count) + ".jsonl")
    dump_jsonl(termID_index, termID_index_path)

    # Corpus statistic
    corpus_stats = {"size": len(doc_index), "non-duplicate": non_duplication_count, "term_count": tokens_count}
    dump_json(corpus_stats, corpus_stats_path)


# Merge two postings of the same term from different partial indexes
def merge_postings(posting1, posting2):
    if posting1 is None:
        return posting2
    if posting2 is None:
        return posting1
    posting1["document"].update(posting2["document"])
    posting1["normalized"].update(posting2["normalized"])
    return posting1


# Process posting (compute idf, create tiers)
def process_posting(posting, corpus_stat):
    posting["idf"] = np.log(corpus_stat["size"]/len(posting["document"]))

    # Tiers sorted by term frequency
    posting["tier"] = list()
    tiers = (50, 100, 1000)
    corpus_stat["tier"] = tiers
    sorted_doc_tf = sorted(posting["document"].items(), key=lambda item: item[1], reverse=True)
    i = 0
    for tier in range(len(tiers)):
        posting["tier"].append([])   # A tier is a set of documents
        for n in range(tiers[tier]):
            if i >= len(sorted_doc_tf):
                break
            posting["tier"][tier].append(sorted_doc_tf[i][0])
            i += 1
    posting["tier"].append([])
    for i in range(i, len(sorted_doc_tf)):  # Put the rest of document to last tier
        posting["tier"][-1].append(sorted_doc_tf[i][0])


def merge_partial_indexes():
    partial_indexes_files = list()
    corpus_stats = load_json(corpus_stats_path)

    print("MERGING PARTIAL INDEXES...")
    for dir_name, subdir_list, file_list in os.walk(index_dir + "\\partial_indexes", topdown=False):
        for f in file_list:
            try:
                partial_indexes_files.append(open(dir_name + "\\" + f, "r"))
            except Exception as e:
                for f_ in partial_indexes_files:
                    f_.close()
                raise e
    if len(partial_indexes_files) == 0:
        print("NO PARTIAL INDEXES FOUND")
        return

    chunk_size = 100000000  # bytes being read in at a time
    merged_index_file = open(merged_index_path, "w")
    term_filepos_index = [None]

    eof_status = np.zeros(len(partial_indexes_files), dtype="?")  # A boolean array for eof status of each partial index
    all_eof = False     # True if all files have reached EOF
    current_term_id = 0
    merged_posting = None
    partial_indexes_chunk = [dict()] * len(partial_indexes_files)
    while not all_eof:  # Keeping going if not all files have reached EOF
        for i in range(len(partial_indexes_files)):
            if partial_indexes_chunk[i] is not None and len(partial_indexes_chunk[i]) == 0:
                partial_indexes_chunk[i] = read_chunk(partial_indexes_files[i], chunk_size)
            if partial_indexes_chunk[i] is None:
                if not eof_status[i]:
                    eof_status[i] = True
                    all_eof = np.prod(eof_status)
                continue
            merged_posting = merge_postings(merged_posting, partial_indexes_chunk[i].pop(current_term_id, None))
        # Write to merged index file, store the file cursor of the term
        if merged_posting is not None:
            process_posting(merged_posting, corpus_stats)
            term_filepos_index[current_term_id] = merged_index_file.tell()
            posting_dump = json.dumps([current_term_id, merged_posting])
            merged_index_file.write(posting_dump)
            merged_index_file.write("\n")
            merged_posting = None
        current_term_id += 1
        term_filepos_index.append(None)

    print("MERGED %i PARTIAL INDEXES WITH %i TERMS" % (len(partial_indexes_files), current_term_id-1))
    dump_json(term_filepos_index, term_filepos_index_path)
    dump_json(corpus_stats, corpus_stats_path)
    # Close Files
    for f in partial_indexes_files:
        f.close()
    merged_index_file.close()


def get_term_posting(term_id, term_filepos_index, merged_index_file):
    if term_id > len(term_filepos_index) - 1 or term_filepos_index[term_id] is None:
        raise ValueError(str(term_id) + " DOES NOT EXIST IN term_filepos_index")
        # return None
    merged_index_file.seek(term_filepos_index[term_id])
    line = merged_index_file.readline()
    data = json.loads(line)
    return data[1]


def index_pipeline(doc_dir="DEV"):
    t = time.time()
    initialize_directory(index_dir)
    doc_index = build_doc_index(doc_dir)
    print(time.time() - t, "\n")
    t = time.time()
    build_partial_index(doc_index)
    print(time.time() - t, "\n")
    t = time.time()
    merge_partial_indexes()
    print(time.time() - t, "\n")


if __name__ == "__main__":
    index_pipeline()
