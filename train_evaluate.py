'''
author: Yufen Huang
'''
import tensorflow as tf
import pickle
#import utils
import time
import numpy as np 
import os
import csv
import random
from esim_model import ESIM
from config import Config
from preprocessing import *
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def load_data(data_file):
    '''
        data should be in the format (encoding with utf-8): row_id /t std /t cus /t /t label
        output shuffle cut datalist
    '''
    data_list = read_file(data_file)
    #shuffle
    random.shuffle(data_list)
    q1_list = [cut_words(l[1]) for l in data_list]
    q2_list = [cut_words(l[2]) for l in data_list]
    label_list = [int(l[3]) for l in data_list]
    #print data information
    print('finish loading data from %s...'%data_file)
    return q1_list, q2_list, label_list

def tokenizer(vocab_file, std_list, cus_list):
    #build or load vocab dict
    if not os.path.exists(vocab_file):
        word2id = build_vocab([i for std in std_list for i in std.split()]+[i for cus in cus_list for i in cus.split()])
        ff = open(vocab_file, 'wb')
        pickle.dump(word2id, ff)
    else:
        ff = open(vocab_file, 'rb')
        word2id = pickle.load(ff)
    config.vocab_size = len(word2id)
    print(len(word2id))
    std_token = []
    for std in std_list:
        token = [0] * config.max_step
        ss = std.split()[:config.max_step]
        token[:len(ss)] = [word2id.get(i,1)for i in ss]
        std_token.append(token)
    cus_token = []
    for cus in cus_list:
        token = [0] * config.max_step
        ss = cus.split()[:config.max_step]
        token[:len(ss)] = [word2id.get(i,1)for i in ss]
        cus_token.append(token)
    return std_token, cus_token

def train(config):

    #load data
    std_list, cus_list, label_list = load_data('atec_nlp_sim_train.csv')
    #print(std_list[:10],cus_list[:10])
    #tokenizer
    std_token, cus_token = tokenizer(config.vocab_file, std_list, cus_list)
    print(std_token[:10],cus_token[:10])
    #train model
    
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        model = ESIM(config)
        sess.run(tf.global_variables_initializer())
        train_sample = list(zip(std_token, cus_token, label_list))
        best_valid_loss = 1000
        train_batch_num = math.floor(len(train_sample)/config.batch_size)
        for epoch in range(config.max_epoches):
            time_start = time.time()
            train_loss = []
            train_acc = []
            for iter_num in range(train_batch_num):
                print('%d/%d'%(iter_num,train_batch_num))
                batch = train_sample[iter_num*config.batch_size:(iter_num+1)*config.batch_size]
                loss, acc = model.train(sess, batch, config)
                train_loss.append(loss)
                train_acc.append(acc)
                #print('=====iter:%d/%d, loss is %.5f'%(iter_num,len(batch_data_train),loss))
            print('Epoch:%2d, training loss is %.5f, acc is %.5f'%(epoch,np.mean(train_loss),np.mean(train_acc)))
            '''
            #evalidation
            valid_loss = []
            valid_acc = []
            for iter_num in range(len(batch_data_valid)):
                loss, acc = model.evaluate(sess, batch_data_valid[iter_num], train_params)
                valid_loss.append(loss)
                valid_acc.append(acc)
            '''
            time_end = time.time()
            epoch_time = time_end - time_start
            #print('          validation loss is %.5f, acc is %.5f'%(np.mean(valid_loss),np.mean(valid_acc)))
            print('training one epoch cost %.6f second !!\n' % epoch_time)
            #save model
            #valid_loss = np.mean(valid_loss)
            #if vaild_loss < best_valid_loss:
            #    best_valid_loss = vaild_loss
            #    utils.save_model(sess, model, train_params)
    
    return

def evalidation(train_params, logger):

    #load data

    #tokenizer

    #load model

    #evalidation

    return

def prediction(train_params, logger):

    #load data

    #tokenizer

    #load model

    #prediction

    return

def inference(std, cus):

    #load model

    #tokenizer

    #prediction

    return

if __name__ == "__main__":
    mode = 1
    std = 'hi'
    cus = 'hi'
    mode = {
        1: 'train',
        2: 'evaluation',
        3: 'prediction',
        4: 'inference'
    }[mode]
    #def train parameters
    config = Config()
    
    '''
    #logger
    log_path = train_params['log_path']
    log_name = train_params['log_name'] + mode
    log = '/'.join([log_path, log_name])
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    elif os.path.exists(log):
        os.remove(log)
    logger = utils.get_logger(log)
    '''
    #call function
    if mode == 'train':
        train(config)
    elif mode == 'evalidation':
        evalidation(config)
    elif mode == 'prediction':
        prediction(config)
    elif mode == 'inference':
        inference(std, cus)


    


