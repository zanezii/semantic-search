#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Author   ：Zane
# @Mail     : zanezii@foxmail.com
# @Date     ：2023/7/1 16:29 
# @File     ：semantic search.py
# @Description :
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from transformers import AutoTokenizer, AutoModel

from typing import Type, List, Dict, Union, Tuple
from Index_construction import IndexConstruction
from embedding import read_config_semantic_info
from utils._utils import serialization,read_json_file


import logging
import subprocess
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# TODO: 读数据
path = '/home/src/'
datasets = 'quora'
path_datasets = f"{path}datasets/{datasets}"
path_datasets_ckpt = f"{path}datasets/{datasets}/ckpt/"

corpus, queries, qrels = GenericDataLoader(data_folder=path_datasets).load(split="test")  # unzipped
serialization(qrels, f'{path_datasets_ckpt}qrels')
print(subprocess.check_output(["pwd"]))
batch_size, is_show_progress_bar, is_convert_to_tensor = read_config_semantic_info()

# TODO: 加载模型
from transformers import AutoTokenizer, AutoModel

# model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
model_ckpt = "sentence-transformers/msmarco-distilbert-base-tas-b"
model = DRES(models.SentenceBERT(model_ckpt), batch_size=batch_size).model
# model_ckpt = "sentence-transformers/msmarco-distilbert-base-tas-b"
# tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# model = AutoModel.from_pretrained(model_ckpt)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# model.to(device)


# TODO: 词嵌入
def get_queries_text_from_dataset(model, queries:Dict[str, str]):
    logging.info("Extracting Queries...")
    query_ids = list(queries.keys())
    results = {qid: {} for qid in query_ids}
    queries = [queries[qid] for qid in queries]

    logging.info("Embedding Queries...")
    queries_embedding = model.encode_queries(queries, batch_size=batch_size, show_progress_bar=is_show_progress_bar
                                             ,convert_to_tensor=is_convert_to_tensor)
    serialization(queries_embedding,f'{path_datasets_ckpt}queries_embedding')
    serialization(queries_embedding,f'{path_datasets_ckpt}query_ids')
    serialization(queries_embedding,f'{path_datasets_ckpt}results')
    return queries_embedding, query_ids, results


def get_corpus_text_from_dataset(model, corpus:Dict[str, Dict[str, str]]):
    logging.info("Extracting Corpus...")
    corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
    corpus = [corpus[cid] for cid in corpus_ids]

    logging.info("Embedding Corpus...")
    corpus_embedding = model.encode_corpus(corpus, batch_size=batch_size, show_progress_bar=is_show_progress_bar,
                                           convert_to_tensor=is_convert_to_tensor)
    serialization(corpus,f'{path_datasets_ckpt}corpus_embedding')
    serialization(corpus_ids,f'{path_datasets_ckpt}corpus_ids')
    return corpus_embedding, corpus_ids


# retriever = EvaluateRetrieval(model, score_function="dot") # or "cos_sim" for cosine similarity
xq, query_ids, rst = get_queries_text_from_dataset(model, queries)
xb, corpus_ids = get_corpus_text_from_dataset(model, corpus)
print(xb.shape, xq.shape)

# TODO: 构建索引
# embeddings_dataset.add_faiss_index(column="embeddings")
dimension = len(xb[0])
num_dataset = len(xb)

ic = IndexConstruction(dimension, nlist=int(num_dataset ** 0.5))
ic.train(xb)
ic.add(xb)

# TODO: 查询并计算得分
K = [1,3,5,10,100,1000]
top_k_values, top_k_idx = ic.search(xq, max(K), nprobe=10)
top_k_values = top_k_values.tolist()
top_k_idx = top_k_idx.tolist()
for query_itr in range(len(xq)):
    query_id = query_ids[query_itr]
    for sub_corpus_id, score in zip(top_k_idx[query_itr], top_k_values[query_itr]):
        corpus_id = corpus_ids[sub_corpus_id]
        if corpus_id != query_id:
            rst[query_id][corpus_id] = score

# TODO: 查看查询结果
# ann.evaluate(question_embedding,,k)
#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rst, K)
# print("---------------------Question---------------------")
# print(question)
# for d, i in zip(D.squeeze(), I.squeeze()):
#     print(f"-----------Distance：{d}-----------")
#     print("[TITLE]")
#     print(comments_dataset["title"][i])
#     print("[BODY]")
#     print(comments_dataset["body"][i])
#     print("[COMMENTS]")
#     print(comments_dataset["comments"][i])
