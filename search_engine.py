import sys
from indexer import index_pipeline
from retriever import search_pipeline

default_dir = "DEV"
top_k = 20

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        job = sys.argv[1]
        if len(sys.argv) == 2:
            doc_dir = default_dir
        else:
            doc_dir = sys.argv[2]
        if job.lower() == "s":
            search_pipeline(top_k)
        elif job.lower() == "b":
            index_pipeline(doc_dir)
    else:
        print("No argument for indexing or searching")
