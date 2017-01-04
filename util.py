# -*- coding: utf-8 -*-
import numpy as np
import MeCab

def get_batch(W,E,P,batch_size):
    size = W.shape[0]
    indexes = np.random.choice(size,batch_size)
    return W[indexes][:],E[indexes][:],P[indexes][:]

def encode_id(x,x2id,size):
    if x not in x2id or x2id[x] > size:
        return x2id["$UNK"]
    else:
        return x2id[x]
    
def fill_pad(l_id,pad_id,length):
    result =  l_id[:min(len(l_id),length)] + [pad_id] * max(0,length - len(l_id))
    return result

def decode(W,E,P,id2word,id2emoji):
    words = []
    for i in range(W.shape[0]):
        words.append([id2word[each] for each in W[i,:]] + [u""])
        for p,j in zip(P[i,:],E[i,:]):
            if p != 0 and j != 0:
                if words[i][p-1] != u"$PAD":
                    words[i][p-1] = id2emoji[j] + u" " +words[i][p-1]
        print u" ".join(filter( lambda w: w != u"$PAD",words[i]))

def encode(sentence,word2id):
    input_instance = []
    parse = MeCab.Tagger("-Owakati").parse
    word_list = parse(sentence.encode("utf-8")).split()
    for w in word_list:
        input_instance.append(unicode(w))
    temp = [encode_id(each,word2id,200000) for each in input_instance]
    W = np.array([fill_pad(temp,0,60)])
    return W
