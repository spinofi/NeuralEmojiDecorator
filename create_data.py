# -*- coding: utf-8 -*-

import numpy as np
import json as js
from sklearn.externals import joblib
import sys
from util import fill_pad, encode_id

input_length = 60
output_length = 20
voc_size = 200000
emoji_size = 750

W = []
E = []
P = []

dicts = joblib.load(sys.argv[2])

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

        temp = [encode_id(each,dicts["word2id"],voc_size) for each in words]
        W.append(fill_pad(temp,0,input_length))

        temp = [dicts["emoji2id"][each] for each in emojis]
        E.append(fill_pad(temp,0,output_length))

        P.append(fill_pad(positions,0,output_length))

joblib.dump({"W":np.array(W),"E":np.array(E),"P":np.array(P),"voc_size":voc_size,"emoji_size":emoji_size,"position_size":input_length+2},"data.pkl")
