import json
from json import dumps
import copy
import pandas as pd
import sys
from elasticsearch import Elasticsearch
import elasticsearch.helpers
from elasticsearch.helpers import parallel_bulk


def main():
    chunk_size = 512
    overlap = 128
    window = chunk_size - overlap

    # print(f'Chunk size: {chunk_size}, overlap: {overlap}, window: {window}'.format(chunk_size, overlap, window))

    # num_of_blocks = 0
    # for i in range(0, len(document), window):
    #     start = i
    #     end = i + chunk_size
    #     num_of_blocks += 1
    #     print(f'Range [{start} : {end}]'.format(start, end))
        # print(document[start:end])

    # print("number of blocks: ", num_of_blocks)

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
    with open("documents.json", "wt", encoding="UTF-8") as f:
        for doc in docs:
            if count % es_chunk_size == 0:
                print(".", flush=True, end="")
            if bulk_ingest:
                json.dump({"index": {"_index": doc["_index"]}}, f)
                f.write("\n")

            for i in range(0, len(doc['_source']['body_content']), window):
                start = i
                end = i + chunk_size
                # print(f'Range [{start} : {end}]'.format(start, end))
                newdoc = copy.deepcopy(doc)
                newdoc['_source']['body_content_window'] = (doc['_source']['body_content'])[start:end]
                del newdoc['_source']['body_content']
                json.dump(newdoc["_source"], f)
                f.write("\n")

            sys.stdout.flush()
            count += 1
    print("\nDone")

    blogs = pd.read_json('documents.json', lines=True)

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
