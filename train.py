# -*- coding:utf-8 -*-
from sklearn.externals import joblib
import argparse
from util import get_batch, decode
from model import Model


dataset = joblib.load("data.pkl")
dicts   = joblib.load("dicts.pkl")
W, E, P = dataset["W"], dataset["E"], dataset["P"]
W_train, E_train, P_train = W[:9000000], E[:9000000], P[:9000000]
W_valid, E_valid, P_valid = W[9000000:], E[9000000:], P[9000000:]
model = Model(input_length=60,output_length=8,w_dim=100,e_dim=80,p_dim=20,enc_dim=100,dec_dim=100,w_size=200000,e_size=752,p_size=62)
#model.load("model.ckpt")

loss_train = 0.
for i in range(100000):
    W_b, E_b, P_b = get_batch(W_train,E_train,P_train,1000)
    loss_train += model.train(W_b,E_b,P_b)
    
    if i % 500 == 0:
        loss_valid = 0.
        for j in range(10):
            W_b, E_b, P_b = get_batch(W_valid,E_valid,P_valid,1000)
            loss_valid += model.error(W_b,E_b,P_b)
            print "epoch:" , str(i/500)
            print "train:",loss_train/500.
            print "valid:",loss_valid/10.
            loss_train = 0.
            print "--- true ----"
            W_b, E_b, P_b = get_batch(W_valid,E_valid,P_valid,1)
            E_sample, P_sample = model.sample(W_b)
            decode(W_b,E_b,P_b,dicts["id2word"],dicts["id2emoji"])
            print "--- sample ---"
            decode(W_b,E_sample,P_sample,dicts["id2word"],dicts["id2emoji"])
            print "---"*20
            model.save("model.ckpt")
                                                                                                                                                                                                                                                                                    
