#!/usr/bin/env python
# encoding: utf-8
"""
File Description: prediction online
Author: nghuyong
Mail: nghuyong@163.com
Created Time: 2019-09-22 17:03
"""
import tokenization
from extract_features import InputExample, convert_examples_to_features
import numpy as np
import requests
import os
import time

vocab_file = os.environ.get('vocab_file', './models/chinese_L-12_H-768_A-12/vocab.txt')
max_token_len = os.environ.get('max_token_len', 128)


def preprocess(text):
    text_a = text
    example = InputExample(unique_id=None, text_a=text_a, text_b=None)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    feature = convert_examples_to_features([example], max_token_len, tokenizer)[0]
    input_ids = np.reshape([feature.input_ids], (1, max_token_len))
    return {
        "inputs": {"input_ids": input_ids.tolist()}
    }


if __name__ == '__main__':
    while True:
        text = input("Input test sentence:\n")
        start = time.time()
        resp = requests.post('http://localhost:8501/v1/models/chnsenticorp:predict', json=preprocess(text))
        end = time.time()
        pro_0, pro_1 = resp.json()['outputs'][0]
        print(f"negative pro:{pro_0} positive pro:{pro_1} time consuming:{int((end - start) * 1000)}ms")
