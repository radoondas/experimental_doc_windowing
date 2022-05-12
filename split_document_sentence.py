import json
from json import dumps
import copy
import pandas as pd
import sys
from elasticsearch import Elasticsearch
import elasticsearch.helpers
from elasticsearch.helpers import parallel_bulk
import nltk
import logging


def main():
    # logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
    chunk_size = 512
    # testing nltk

    es_host = "http://localhost:9200"
    es_user = "elastic"
    es_password = "password"
    es_timeout = 120
    es_chunk_size = 100
    bulk_ingest = False

    index_name=".ent-search-engine-documents-elastic-blog"

    es = Elasticsearch(hosts=[es_host], basic_auth=(es_user, es_password),
                           verify_certs=True, request_timeout=es_timeout)

    # docs = elasticsearch.helpers.scan(es, query={ "query": { "match": { "id": "620b140a7053e5cfab51acda" } } },
    docs = elasticsearch.helpers.scan(es, query={"query": {"match_all": {}}},
                                      size=es_chunk_size, request_timeout=es_timeout, index=index_name)

    count = 0
    with open("documents-sentence.json", "wt", encoding="UTF-8") as f:
        for doc in docs:
            if count % es_chunk_size == 0:
                print(".", flush=True, end="")
            if bulk_ingest:
                json.dump({"index": {"_index": doc["_index"]}}, f)
                f.write("\n")

            sentences = nltk.tokenize.sent_tokenize(doc['_source']['body_content'])
            logging.debug('Number of sentences: {}'.format(len(sentences)))
            logging.debug('Sentences: '.format(sentences))

            next_window = True
            number_of_sentences = len(sentences)
            current_position = 0

            while next_window:
                full_window = False
                window_len = 0
                window_num_of_sentences = 0
                window_text = ""
                while not full_window:
                    if current_position == number_of_sentences:
                        full_window = True
                        next_window = False
                    else:
                        # print('Current sentence position: {}'.format(current_position))
                        logging.debug('Current sentence position: {}'.format(current_position))
                        # print('Old window length: {}'.format(window_len))
                        logging.debug('Old window length: {}'.format(window_len))
                        # print('Sentence length: {}'.format(len(sentences[current_position])))
                        logging.debug('Sentence length: {}'.format(len(sentences[current_position])))
                        window_len += len(sentences[current_position])
                        # print('New window length: {}'.format(window_len))
                        logging.debug('New window length: {}'.format(window_len))
                        # print('__Current sentence: {}'.format(sentences[current_position]))
                        logging.debug('__Current sentence: {}'.format(sentences[current_position]))

                        window_text += " " + sentences[current_position]
                        window_num_of_sentences += 1
                        # print('Number of sentences: {}'.format(window_num_of_sentences))
                        logging.debug('Number of sentences: {}'.format(window_num_of_sentences))
                        # print(window_text)
                        logging.debug('Full window text: {}'.format(window_text))
                        if window_len > chunk_size:
                            full_window = True
                            if window_num_of_sentences == 1:
                                current_position += 1
                            # write the document
                            newdoc = copy.deepcopy(doc)
                            newdoc['_source']['body_content_window'] = window_text
                            del newdoc['_source']['body_content']
                            json.dump(newdoc["_source"], f)
                            f.write("\n")
                        else:
                            current_position += 1
                # print("DONE")
                logging.debug("DONE")

            sys.stdout.flush()
            count += 1
    print("\nDone")

    blogs = pd.read_json('documents-sentence.json', lines=True)

    count = 0
    for success, info in parallel_bulk(
            client=es,
            actions=gen_rows(blogs),
            thread_count=4,
            chunk_size=es_chunk_size,
            timeout='%ss' % es_timeout,
            index='chunked-blogs'
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


def gen_rows(df):
    df = df.fillna("")
    for doc in df.to_dict(orient='records'):
        yield doc


if __name__ == '__main__':
    main()
