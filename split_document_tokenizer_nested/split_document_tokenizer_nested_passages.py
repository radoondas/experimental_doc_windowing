import json
import pandas as pd
import sys
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk
import nltk
from transformers import AutoTokenizer
import ndjson

doc_file = 'documents-out.json'
source_doc_file = 'documents-in.json'
blogs_windows_index = 'blogs-windows'
blogs_windows_mapping = 'blogs-windows-mappings.json'

es_host = "http://localhost:9200"
es_user = "elastic"
es_password = "changeme"

es_timeout = 120
es_chunk_size = 100
bulk_ingest = False

max_num_of_tokens = 512


def main():
    # Connects to ES and dump the content of the index and split textfield (doc['_source']['body_content'])
    # into sliding windows.
    # Save the content (new documents) into the file `doc_file`
    # Works with index from ES webcrawler in ESS
    create_sliding_windows()

    # Indexes output (json file) from the above step into the ES
    index_blogs_windows()


def gen_rows(df):
    df = df.fillna("")
    for doc in df.to_dict(orient='records'):
        yield doc


# We take the document's text field we want to split into multiple. We store split text in the same document in the field
# 'passades' as nested documents
# Split the text into sentences using sentence tokenizer
# Take sentence and split into tokens using tokenizer. Count number of tokens in the sentence.
# if number of tokens is less than 512, save sentence into buffer, and take the next sentence.
# Count the number of tokens together with number of tokens from previous sentences and if less than 512, continue with
# the next sentence.
# If the number of sentences is >= 512, save the document with the previous sentence's buffer. Discard current sentence.
# Take the last saved sentence from previous window and start from there for the new window.
# With the approach we cerate ovelapping text windws with one sentence.
def create_sliding_windows():
    # es = Elasticsearch(hosts=[es_host])
    es = Elasticsearch(hosts=[es_host], basic_auth=(es_user, es_password),
                       verify_certs=True, request_timeout=es_timeout)


    with open(source_doc_file) as f_source:
        docs = ndjson.load(f_source)

    # https://huggingface.co/sentence-transformers/msmarco-MiniLM-L-12-v3
    tokenizer_v3 = AutoTokenizer.from_pretrained('sentence-transformers/msmarco-MiniLM-L-12-v3')

    count = 0
    with open(doc_file, "wt", encoding="UTF-8") as f:
        # for each original document
        for doc in docs:
            if count % es_chunk_size == 0:
                print(".", flush=True, end="")
            if bulk_ingest:
                json.dump({"index": {"_index": doc["_index"]}}, f)
                f.write("\n")

            # Split text from the document into sentences
            sentences = nltk.tokenize.sent_tokenize(doc['body_content'])
            print('')
            print('Number of sentences: {}'.format(len(sentences)))
            print('Sentences: {}'.format(sentences))
            print('Starting sliding window definition ...')

            next_window = True
            number_of_sentences = len(sentences)
            current_position = 0

            # Define nested field for the passages
            doc['passages'] = []
            while next_window:
                full_window = False
                window_num_of_sentences = 0
                window_text = ""
                window_num_of_tokens = 0

                while not full_window:
                    if current_position == number_of_sentences:
                        full_window = True
                        next_window = False
                    else:
                        print('Current sentence position: {}'.format(current_position))
                        print('Current sentence: {}'.format(sentences[current_position]))

                        print('Old window length (tokens): {}'.format(window_num_of_tokens))

                        # Get sentence tokens
                        tokens_v3 = tokenizer_v3.tokenize(sentences[current_position], padding=True, truncation=True,
                                                          return_tensors='pt')
                        print('Tokens: ', tokens_v3)
                        num_of_tokens = len(tokens_v3)
                        print('Current sentence token length: {}'.format(num_of_tokens))

                        if (window_num_of_tokens + num_of_tokens) <= max_num_of_tokens:
                            window_num_of_tokens += num_of_tokens
                            print('New window number of tokens: {}'.format(window_num_of_tokens))

                            window_text += " " + sentences[current_position]
                            window_num_of_sentences += 1

                            print('Number of sentences: {}'.format(window_num_of_sentences))
                            current_position += 1
                        else:
                            print('Number of tokens in the sentence will create window over 512 tokens. '
                                  'Sliding window is full.')
                            full_window = True
                            if window_num_of_sentences == 0:
                                print('This is very long sentence and needs separate window.')
                                window_text += " " + sentences[current_position]
                                window_num_of_sentences += 1
                                current_position += 1

                            doc['passages'].append(window_text)

                        print('Full window text: {}'.format(window_text))
                print("------------ Window: DONE")
            json.dump(doc, f)
            f.write("\n")

            sys.stdout.flush()
            count += 1
            print("\nDone")
    print("\nDone Done")


def index_blogs_windows():
    # Index documents into ES
    es = Elasticsearch(hosts=[es_host], basic_auth=(es_user, es_password),
                       verify_certs=True, request_timeout=es_timeout)

    with open(blogs_windows_mapping, "r") as config_file_blg:
        config_blg = json.loads(config_file_blg.read())

        if not es.indices.exists(index=blogs_windows_index):
            print("Creating index %s" % blogs_windows_index)
            es.indices.create(index=blogs_windows_index, mappings=config_blg["mappings"],
                              settings=config_blg["settings"], ignore=[400, 404])
        else:
            print("Index " + "blogs_windows_index" + " already exists. Is that OK? Check. I'll index anyways into that index.")

    blogs = pd.read_json(doc_file, lines=True)

    count = 0
    for success, info in parallel_bulk(
            client=es,
            actions=gen_rows(blogs),
            thread_count=4,
            chunk_size=es_chunk_size,
            timeout='%ss' % es_timeout,
            index=blogs_windows_index
    ):
        if success:
            count += 1
            if count % es_chunk_size == 0:
                print('Indexed %s documents' % str(count), flush=True)
                sys.stdout.flush()
        else:
            print('Doc failed', info)

    print('Indexed %s blogs embeddings documents' % str(count), flush=True)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
