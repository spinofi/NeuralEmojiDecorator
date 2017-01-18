# -*- coding:utf-8 -*-
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import argparse
from util import get_batch, decode
from model import Model


dataset = joblib.load("data.pkl")
dicts   = joblib.load("dicts.pkl")
W, E, P = dataset["W"], dataset["E"], dataset["P"]
ts = 7000000
print W.shape
W_train, E_train, P_train = W[:ts], E[:ts], P[:ts]
W_valid, E_valid, P_valid = W[ts:], E[ts:], P[ts:]
model = Model(input_length=60,output_length=8,w_dim=300,e_dim=200,p_dim=100,enc_dim=300,dec_dim=300,w_size=200000,e_size=750,p_size=62)
model.load("model.ckpt")

loss_train = 0.
for i in range(10000000):
    W_b, E_b, P_b = get_batch(W_train,E_train,P_train,1000)
    loss_train += model.train(W_b,E_b,P_b)
    if i % 100 == 0:
        W_b, E_b, P_b = get_batch(W_valid,E_valid,P_valid,1000)
        #W_b, E_b, P_b = get_batch(W_train,E_train,P_train,1000)
        loss_valid = model.error(W_b,E_b,P_b)
        print "epoch:" , str(i/100)
        print "train:",loss_train/100.
        print "valid:",loss_valid
        loss_train = 0.
        print "--- true ----"
        W_b, E_b, P_b = get_batch(W_train,E_train,P_train,1)
        E_sample, P_sample = model.sample(W_b)
        decode(W_b,E_b,P_b,dicts["id2word"],dicts["id2emoji"])
        print "--- sample ---"
        decode(W_b,E_sample,P_sample,dicts["id2word"],dicts["id2emoji"])
        print "---"*20
        model.save("model.ckpt")
                                                                                                                                                                                                                                                                                    
