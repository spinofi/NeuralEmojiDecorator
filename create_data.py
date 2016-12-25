
# -*- coding: utf-8 -*-

import numpy as np
import json as js
from sklearn.externals import joblib
import sys

input_length = 60
output_length = 20
voc_size = 200000
emoji_size = 750

W = []
E = []
P = []

dicts = joblib.load(sys.argv[2])


def encode(x,x2id,size):
    if x not in x2id or x2id[x] > size:
        return x2id["$UNK"]
    else:
        return x2id[x]

def fill_pad(l_id,pad_id,length):
    result =  l_id[:min(len(l_id),length)] + [pad_id] * max(0,length - len(l_id))
    assert(len(result) == length)
    return result

with open(sys.argv[1]) as f:
    for line in f:
        j = js.loads(line)
        words = j["input_instance"]
        targets = j["output_instance"]
        if len(words) > 60:
            continue
        if len(targets) > 20:
            continue

        emojis = [each[0] for each in targets]
        positions = [int(each[1])+1 for each in targets]

        temp = [encode(each,dicts["word2id"],voc_size) for each in words]
        W.append(fill_pad(temp,0,input_length))

        temp = [dicts["emoji2id"][each] for each in emojis]
        E.append(fill_pad(temp,0,output_length))

        P.append(fill_pad(positions,0,output_length))

joblib.dump({"W":np.array(W),"E":np.array(E),"P":np.array(P),"voc_size":voc_size,"emoji_size":emoji_size,"position_size":input_length+2},"data.pkl")
