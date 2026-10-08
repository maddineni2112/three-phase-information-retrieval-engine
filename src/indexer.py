import os
import time
from collections import defaultdict
from nltk.stem import PorterStemmer
from text_parser import load_stopwords, process_documents

def build_forward_index(processed_docs, word_dict):
    forward_index = {}
    for doc_id, tokens in processed_docs.items():
        word_freq = defaultdict(int)
        for token in tokens:
            word_id = str(word_dict[token])
            word_freq[word_id] += 1
        forward_index[doc_id] = dict(word_freq)
    return forward_index

def build_inverted_index(forward_index):
    inverted_index = defaultdict(dict)
    for doc_id, word_freqs in forward_index.items():
        for word_id, freq in word_freqs.items():
            inverted_index[word_id][doc_id] = freq
    return inverted_index

def save_index_formatted(index, filename):
    with open(filename, 'w') as f:
        for word_id in sorted(index, key=lambda x: int(x)):
            postings = sorted(index[word_id].items(), key=lambda x: int(x[0]))
            posting_str = "; ".join(f"{doc_id}: {freq}" for doc_id, freq in postings)
            f.write(f"{word_id}: {posting_str}\n")

def search_term(term, word_dict, inverted_index):
    stemmer = PorterStemmer()
    term = stemmer.stem(term.lower())
    term_id = word_dict.get(term)
    if term_id is None:
        print(f"'{term}' not found in dictionary.")
        return
    term_id_str = str(term_id)
    if term_id_str in inverted_index:
        print(f"Term '{term}' (word ID {term_id_str}) found in:")
        for doc_id in sorted(inverted_index[term_id_str], key=int):
            freq = inverted_index[term_id_str][doc_id]
            print(f"{doc_id}: {freq}", end="; ")
    else:
        print(f"No documents contain the term '{term}'.")

if __name__ == "__main__":
    STOPWORDS_FILE = "stopwordlist.txt"
    DATA_DIR = "ft911/"

    total_start = time.time()

    stopwords = load_stopwords(STOPWORDS_FILE)
    parse_start = time.time()
    _, docs_output, processed_docs, word_dict = process_documents(DATA_DIR, stopwords)
    parse_end = time.time()
    print(f"Parsing completed in {parse_end - parse_start:.2f} seconds.")

    forward_start = time.time()
    forward_index = build_forward_index(processed_docs, word_dict)
    forward_end = time.time()
    print(f"Forward indexing completed in {forward_end - forward_start:.2f} seconds.")

    inverted_start = time.time()
    inverted_index = build_inverted_index(forward_index)
    inverted_end = time.time()
    print(f"Inverted indexing completed in {inverted_end - inverted_start:.2f} seconds.")

    save_start = time.time()
    save_index_formatted(forward_index, "forward_index.txt")
    save_index_formatted(inverted_index, "inverted_index.txt")
    save_end = time.time()
    print(f"Index saving completed in {save_end - save_start:.2f} seconds.")

    total_end = time.time()
    print(f"Total time elapsed: {total_end - total_start:.2f} seconds.")
    print("Formatted indexes saved with original document IDs.")
    # Print sizes of index files
    forward_size = os.path.getsize("forward_index.txt")
    inverted_size = os.path.getsize("inverted_index.txt")
    print(f"Size of forward_index.txt: {forward_size / 1024:.2f} KB")
    print(f"Size of inverted_index.txt: {inverted_size / 1024:.2f} KB")


    # Optional: interactive term search
    while True:
        query = input("\n Enter a term to search in the inverted index (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break
        search_term(query, word_dict, inverted_index)
