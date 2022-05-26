import json
import copy
import pandas as pd
import sys
from elasticsearch import Elasticsearch
import elasticsearch.helpers
from elasticsearch.helpers import parallel_bulk
import nltk
from transformers import AutoTokenizer

doc_file = 'documents.json'
source_blogs_index = ".ent-search-engine-documents-elastic-blogs"
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
    # Works with index from ES webcrawler
    create_sliding_windows()

    # Indexes output (json file) from the above step into the ES
    index_blogs_windows()


def gen_rows(df):
    df = df.fillna("")
    for doc in df.to_dict(orient='records'):
        yield doc


def create_sliding_windows():
    es = Elasticsearch(hosts=[es_host], basic_auth=(es_user, es_password),
                       verify_certs=True, request_timeout=es_timeout)

    # docs = elasticsearch.helpers.scan(es, query={"query": {"match": {"id": "627bf98f49aefa0bf2287ea4"}}},
    docs = elasticsearch.helpers.scan(es, query={"query": {"match_all": {}}},
                                      size=es_chunk_size, request_timeout=es_timeout, index=source_blogs_index)

    # https://huggingface.co/sentence-transformers/msmarco-MiniLM-L-12-v3
    tokenizer_v3 = AutoTokenizer.from_pretrained('sentence-transformers/msmarco-MiniLM-L-12-v3')

    count = 0
    with open(doc_file, "wt", encoding="UTF-8") as f:
        for doc in docs:
            if count % es_chunk_size == 0:
                print(".", flush=True, end="")
            if bulk_ingest:
                json.dump({"index": {"_index": doc["_index"]}}, f)
                f.write("\n")

            sentences = nltk.tokenize.sent_tokenize(doc['_source']['body_content'])
            print('')
            print('Number of sentences: {}'.format(len(sentences)))
            print('Sentences: {}'.format(sentences))
            print('Starting sliding window definition ...')

            next_window = True
            number_of_sentences = len(sentences)
            current_position = 0

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

                            newdoc = copy.deepcopy(doc)
                            newdoc['_source']['body_content_window'] = window_text
                            del newdoc['_source']['body_content']
                            json.dump(newdoc["_source"], f)
                            f.write("\n")

                        print('Full window text: {}'.format(window_text))
                print("------------ Window: DONE")

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
            es.indices.create(index=blogs_windows_index, mappings=config_blg["mappings"], settings=config_blg["settings"],
                      ignore=[400, 404])
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
