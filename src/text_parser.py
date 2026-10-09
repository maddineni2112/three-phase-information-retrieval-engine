import os
import re
import nltk
from nltk.stem import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt')

# uploading the Stopwords
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        stopwords = set(word.strip() for word in f.readlines())
    return stopwords

# Tokenization function
def tokenize(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = re.split(r'\W+', text)  # Split on non-alphanumeric characters
    tokens = [t for t in tokens if t and not any(char.isdigit() for char in t)]  # Remove tokens with numbers
    return tokens

# Process documents and build dictionaries
def process_documents(directory, stopwords):
    word_dict = {}  # Word-to-ID mapping
    doc_dict = {}   # Document-to-ID mapping
    word_id = 1
    doc_id = 0
    processed_docs = {}
    stemmer = PorterStemmer()

    output_tokens = []
    output_docs = []

    # Iterate through all ft911_x files in the subdirectory
    files = [f for f in os.listdir(directory) if f.startswith("ft911_")]
    files.sort (key=lambda x: int(re.search(r'ft911_(\d+)', x).group(1)))
               
    for filename in files:
        if filename.startswith("ft911_"):
            filepath = os.path.join(directory, filename)

            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

                # Extract individual documents using <DOC> tags
                documents = re.findall(r"<DOC>(.*?)</DOC>", content, re.DOTALL)

                for doc in documents:
                    # Extract DOCNO
                    doc_name_match = re.search(r"<DOCNO>(.*?)</DOCNO>", doc)

                    if doc_name_match:
                        doc_name = doc_name_match.group(1).strip()
                        doc_id = int(doc_name.split('-')[1])
                        if doc_name not in doc_dict:
                            doc_dict[doc_name] = doc_id
                            output_docs.append(f"{doc_name}\t{doc_id}")

                    # capturing the text
                    doc_text_match = re.search(r"<TEXT>(.*?)</TEXT>", doc, re.DOTALL)
                    doc_text = doc_text_match.group(1).strip() if doc_text_match else ""
                    # Tokenizing the text
                    tokens = tokenize(doc_text)
                    # Removing the stopwords and stemming
                    filtered_tokens = [stemmer.stem(t) for t in tokens if t not in stopwords]

                    processed_docs[doc_id] = filtered_tokens

                    # Adding filtered tokens to word_dict, ensuring uniqueness
                    for token in filtered_tokens:
                        if token not in word_dict:
                            word_dict[token] = None  #storing the unique tokens, ID's will be assigned later after sorting

    # Sort tokens alphabetically
    word_dict = dict(sorted((key, None) for key in word_dict.keys()))


    # Assign unique IDs to sorted tokens
    for word in word_dict.keys():
        word_dict[word] = word_id
        output_tokens.append(f"{word}\t{word_id}")
        word_id += 1

    return output_tokens, output_docs, processed_docs, word_dict

# Main execution
if __name__ == "__main__":
    STOPWORDS_FILE = "stopwordlist.txt"
    DATA_DIRECTORY = "ft911/"  

    stopwords = load_stopwords(STOPWORDS_FILE)
    tokens_output, docs_output, processed_docs, word_dict = process_documents(DATA_DIRECTORY, stopwords)


    # Save output to file
    with open("parser_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(tokens_output) + "\n\n")
        f.write("\n".join(docs_output))

    print("Parsing complete! Output saved in parser_output.txt")
