# -*- coding: utf-8 -*-
import sys
import json as js
from sklearn.externals import joblib

def count(token,dict):
    if token in dict:
        dict[token] += 1
    else:
        dict[token] = 1

def createX2id(dict):
    keys = dict.keys()
    temp = sorted(keys,key=lambda k: - dict[k])
    x2id = {u"$UNK":1,u"$PAD":0}
    id2x = {1:u"$UNK",0:u"$PAD"}
    i = 2
    for x in temp:
        x2id[x] = i
        id2x[i] = x
        i += 1
    return x2id, id2x


WordDict = {}
EmojiDict = {}

lw,lt = 0,0

with open(sys.argv[1]) as f:
    for line in f:
        j = js.loads(line)
        words = j["input_instance"]
        targets = j["output_instance"]
        if len(words) > 60:
            continue
        if len(targets) > 20:
            continue
        lw = max(lw,len(words))
        lt = max(lt,len(targets))
        for w in words:
            count(w,WordDict)
        for e,p in targets:
            count(e,EmojiDict)

print lw,lt
dicts = {}
dicts["word2id"],dicts["id2word"] = createX2id(WordDict)
dicts["emoji2id"],dicts["id2emoji"] = createX2id(EmojiDict)

joblib.dump(dicts,"dicts.pkl")
        
    
