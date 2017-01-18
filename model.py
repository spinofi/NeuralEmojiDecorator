# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

def for_each_lookup(list_a,embeddings):
    return [tf.nn.embedding_lookup(embeddings,each) for each in list_a]

def linear(x,W,b):
    return tf.matmul(x,W) + b

def weight_variable(shape,name):
    initial = tf.random_normal(shape, stddev = 0.1)
    return tf.get_variable(initializer=initial,name=name)

def bias_variable(shape,name):
    initial= tf.constant(0.0, shape = shape)
    return tf.get_variable(initializer=initial,name=name)

def sample_from_multi(preds):
    return np.array([np.argmax(np.random.multinomial(1,pred * 0.999)) for pred in list(preds)])

class Model:
    def __init__(self,input_length,output_length,w_dim,e_dim,p_dim,enc_dim,dec_dim,w_size,e_size,p_size):
        self.input_length = input_length
        self.output_length = output_length
        self.w_dim = w_dim
        self.e_dim = e_dim
        self.p_dim = p_dim
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.w_size = w_size
        self.e_size = e_size
        self.p_size = p_size
                
        self.word_inputs     = [tf.placeholder(tf.int32,(None,)) for _ in range(input_length)]
        self.position_inputs = [tf.placeholder(tf.int32,(None,)) for _ in range(output_length)]
        self.emoji_inputs    = [tf.placeholder(tf.int32,(None,)) for _ in range(output_length)]

        self.emoji_embeddings    = weight_variable((e_size,e_dim),"emoji_embeddings")
        self.position_embeddings = weight_variable((p_size,p_dim),"position_embeddings")
        self.word_embeddings     = weight_variable((w_size,w_dim),"word_embeddings")

        self.word_input_embeddings     = for_each_lookup(self.word_inputs, self.word_embeddings)
        temp                           = for_each_lookup(self.emoji_inputs, self.emoji_embeddings)
        self.emoji_input_embeddings    = [tf.zeros_like(temp[0])] + temp[:-1]
        temp                           = for_each_lookup(self.position_inputs, self.position_embeddings)
        self.position_input_embeddings = [tf.zeros_like(temp[0])] + temp[:-1]

        self.emoji_W = weight_variable((self.dec_dim,self.e_size),"emoji_W")
        self.emoji_b = bias_variable((self.e_size,),"emoji_b")
        self.position_W = weight_variable((self.dec_dim,self.p_size),"position_W")
        self.position_b = bias_variable((self.p_size,),"position_b")
                                
        self.encoder_cell = tf.nn.rnn_cell.GRUCell(self.enc_dim)
        self.decoder_cell = tf.nn.rnn_cell.GRUCell(self.dec_dim)

        # encoder rnn loop
        with tf.variable_scope("encoder") as scope:
            state = tf.zeros_like(self.word_input_embeddings[0])
            for i in range(self.input_length):
                input = self.word_input_embeddings[i]
                _, state = self.encoder_cell(input, state)
                scope.reuse_variables()
            self.encoder_final_state = state

        # decoder rnn loop
        self.decoder_emoji_logits = []
        self.decoder_position_logits = []
        with tf.variable_scope("decoder") as scope:
            state = self.encoder_final_state
            for i in range(self.output_length):
                input = tf.concat(1,[self.emoji_input_embeddings[i], self.position_input_embeddings[i]])
                output, state = self.decoder_cell(input, state)
                emoji_logit = linear(output, self.emoji_W, self.emoji_b)
                position_logit = linear(output, self.position_W, self.position_b)
                self.decoder_emoji_logits.append(emoji_logit)
                self.decoder_position_logits.append(position_logit)
                scope.reuse_variables()
                                
            self.decoder_emoji_distributions = [tf.nn.softmax(each) for each in self.decoder_emoji_logits]
            self.decoder_position_distributions = [tf.nn.softmax(each) for each in self.decoder_position_logits]

        # loss
        weights = [tf.ones_like(self.emoji_inputs[0],dtype=tf.float32) for each in range(self.output_length)]
        self.emoji_loss = tf.nn.seq2seq.sequence_loss(self.decoder_emoji_logits, self.emoji_inputs, weights)
        self.position_loss = tf.nn.seq2seq.sequence_loss(self.decoder_position_logits, self.position_inputs, weights)
        self.loss = self.emoji_loss + self.position_loss

        # optim
        self.optim = tf.train.AdamOptimizer(0.003).minimize(self.loss)

        # session
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

        # saver

        self.saver = tf.train.Saver()


    def train(self,W,E,P):
        feed = {}
        batch_size = W.shape[0]
        for i in range(self.input_length):
            feed[self.word_inputs[i]] = W[:,i]
        for i in range(self.output_length):
            feed[self.emoji_inputs[i]] = E[:,i]
            feed[self.position_inputs[i]] = P[:,i]
        loss, _ = self.session.run([self.loss,self.optim],feed_dict = feed)
        return loss
    
    def error(self,W,E,P):
        feed = {}
        batch_size = W.shape[0]
        for i in range(self.input_length):
            feed[self.word_inputs[i]] = W[:,i]
        for i in range(self.output_length):
            feed[self.emoji_inputs[i]] = E[:,i]
            feed[self.position_inputs[i]] = P[:,i]
        loss = self.session.run(self.loss,feed_dict = feed)
        return loss

    def sample(self,W):
        feed = {}
        batch_size = W.shape[0]
        for i in range(self.input_length):
            feed[self.word_inputs[i]] = W[:,i]
        feed[self.emoji_inputs[0]] = np.zeros((batch_size,))
        feed[self.position_inputs[0]] = np.zeros((batch_size,))
        E_sample = []
        P_sample = []
        for i in range(self.output_length):
            preds = self.session.run(self.decoder_emoji_distributions[i],feed_dict=feed)
            if i < 8:
                #preds[0,0] = 0
                emoji_output = sample_from_multi(preds)
            else:
                emoji_output = np.argmax(preds,axis=1)
            preds = self.session.run(self.decoder_position_distributions[i],feed_dict=feed)
            
            if i < 8:
                #preds[0,0] = 0
                position_output = sample_from_multi(preds)
            else:
                position_output = np.argmax(preds,axis=1)
            feed[self.emoji_inputs[i]] = emoji_output
            feed[self.position_inputs[i]] = position_output
            E_sample.append(emoji_output)
            P_sample.append(position_output)
        return np.array(E_sample).T, np.array(P_sample).T


    def save(self,name):
        self.saver.save(self.session,name)
        print "model saved"

    def load(self,name):
        self.saver.restore(self.session,name)
        print "model restored"

