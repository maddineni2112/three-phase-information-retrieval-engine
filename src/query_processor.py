import re
import math
from collections import defaultdict, Counter
import nltk
from nltk.stem import PorterStemmer
from text_parser import load_stopwords, tokenize, process_documents
from indexer import build_forward_index, build_inverted_index

nltk.download('punkt')

STOPWORDS_FILE = "stopwordlist.txt"
TOPIC_FILE = "topics.txt"
QRELS_FILE = "main.qrels"
DATA_DIRECTORY = "ft911/"
OUTPUT_FILE = "vsm_output.txt"

def parse_topics(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    queries = []
    topics = re.findall(r"<top>(.*?)</top>", content, re.DOTALL)
    for top in topics:
        topic_id = re.search(r"<num> Number:\s*(\d+)", top).group(1)
        title = re.search(r"<title>\s*(.*)", top).group(1).strip()
        desc = re.search(r"<desc> Description:\s*(.*?)\n\n", top, re.DOTALL)
        narr = re.search(r"<narr> Narrative:\s*(.*)", top, re.DOTALL)
        desc_text = desc.group(1).strip().replace("\n", " ") if desc else ""
        narr_text = narr.group(1).strip().replace("\n", " ") if narr else ""
        queries.append((topic_id, title, title + " " + desc_text, title + " " + desc_text + " " + narr_text))
    return queries

def load_qrels(filename):
    qrels = defaultdict(set)
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            topic_id, _, doc_name, relevance = line.strip().split()
            if relevance == "1":
                qrels[topic_id].add(int(doc_name.split('-')[1]))  # convert FT911-10000 -> 10000
    return qrels

def compute_idf(inverted_index, total_docs):
    return {term: math.log(total_docs / len(postings), 10) if len(postings) else 0
            for term, postings in inverted_index.items()}

def compute_tf_idf(vec, idf):
    return {term: (1 + math.log(freq, 10)) * idf.get(term, 0) for term, freq in vec.items()}

def cosine_similarity(query_vec, doc_vec):
    dot = sum(query_vec[t] * doc_vec.get(t, 0) for t in query_vec)
    query_mag = math.sqrt(sum(v ** 2 for v in query_vec.values()))
    doc_mag = math.sqrt(sum(v ** 2 for v in doc_vec.values()))
    return dot / (query_mag * doc_mag) if query_mag > 0 and doc_mag > 0 else 0.0

def evaluate(retrieved_docs, relevant_docs):
    if not retrieved_docs:
        return 0.0, 0.0
    retrieved_set = set(retrieved_docs)
    relevant_retrieved = retrieved_set.intersection(relevant_docs)
    precision = len(relevant_retrieved) / len(retrieved_docs)
    recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0
    return precision, recall

def run_query_processor():
    stopwords = load_stopwords(STOPWORDS_FILE)
    tokens_output, docs_output, processed_docs, word_dict = process_documents(DATA_DIRECTORY, stopwords)
    doc_id_to_name = {int(line.split("\t")[1]): line.split("\t")[0] for line in docs_output}

    forward_index = build_forward_index(processed_docs, word_dict)
    inverted_index = build_inverted_index(forward_index)
    idf = compute_idf(inverted_index, len(processed_docs))
    queries = parse_topics(TOPIC_FILE)
    qrels = load_qrels(QRELS_FILE)
    stemmer = PorterStemmer()
    comparison_results = defaultdict(dict)

    with open(OUTPUT_FILE, 'w') as f:
        for topic_id, title, title_desc, full_query in queries:
            for setting_name, query_text in zip(['title', 'title+desc', 'title+desc+narr'], [title, title_desc, full_query]):
                tokens = tokenize(query_text)
                query_terms = [stemmer.stem(t) for t in tokens if t not in stopwords and t in word_dict]
                query_tf = Counter()
                weight = {'title': 3, 'title+desc': 2, 'title+desc+narr': 1}[setting_name]
                for t in query_terms:
                    query_tf[str(word_dict[t])] += weight

                query_vec = compute_tf_idf(query_tf, idf)
                scores = {}
                for term in query_vec:
                    if term in inverted_index:
                        for doc_id in inverted_index[term]:
                            if doc_id not in scores:
                                doc_vec = compute_tf_idf(forward_index[int(doc_id)], idf)
                                scores[doc_id] = cosine_similarity(query_vec, doc_vec)

                ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top_docs = [int(doc_id) for doc_id, _ in ranked_docs[:1000]]
                for rank, (doc_id, score) in enumerate(ranked_docs[:1000], 1):
                    f.write(f"{topic_id}\t{doc_id_to_name[int(doc_id)]}\t{rank}\t{score:.6f}\n")

                # Print Top 10 Results
                print(f"\n🔎 Topic {topic_id} | Setting: {setting_name}")
                for rank, (doc_id, score) in enumerate(ranked_docs[:10], 1):
                    print(f"{rank}. {doc_id_to_name[int(doc_id)]} — score: {score:.4f}")

                precision, recall = evaluate(top_docs, qrels.get(topic_id, set()))
                comparison_results[topic_id][setting_name] = (precision, recall)

    # Summary Table
    print("\n=== Precision and Recall Comparison ===")
    print(f"{'Topic':<6} {'Setting':<20} {'Precision':<10} {'Recall':<10}")
    print("-" * 50)
    for topic_id in sorted(comparison_results):
        for setting in ['title', 'title+desc', 'title+desc+narr']:
            prec, rec = comparison_results[topic_id].get(setting, (0.0, 0.0))
            print(f"{topic_id:<6} {setting:<20} {prec:<10.4f} {rec:<10.4f}")

if __name__ == "__main__":
    run_query_processor()
