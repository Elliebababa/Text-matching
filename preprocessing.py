'''
@author: Yufen Huang
'''
#preprocessing
import jieba
import re
import pickle

def read_file(input_file):
    data_list = []
    output_file  = input_file.replace('.csv','')
    output_file += '_short_pairs_0'
    with open(input_file, 'r', encoding='UTF-8') as f, open(output_file,'w', encoding='UTF-8') as f1:
        for line in f:
            l = line.split('\t')
            if len(l[1]) > 50 or len(l[2]) > 50 or (int(l[3]) == 1):
                continue
            l[1] = remove_pun(l[1])#l[1].replace(' ','')
            l[2] = remove_pun(l[2])#l[2].replace(' ','')
            l[3] = l[3].replace('\n','')
            data_list.append(l)
            f1.write('\t'.join(l[1:])+'\n')
    return data_list

def remove_pun(text):
    #去掉中文标点
    pun_chi  = re.compile(u"[\u3002|\uFF1F|\uFF01|\uFF0C|\u3001|\uFF1B|\uFF1A|\u300C|\u300D|\u300E|\u300F|\u2018|\u2019|\u201C|\u201D|\uFF08|\uFF09|\u3014|\u3015|\u3010|\u3011|\u2014|\u2026|\u2013|\uFF0E|\u300A|\u300B|\u3008|\u3009]+")
    text = re.sub(pun_chi, '', text)
    #英文标点
    pun_eng = re.compile(u"[\!\@\#\$\%\^\&\*\(\)\-\=\[\]\{\}\\\|\;\'\:\"\,\.\/\<\>\?\/\*\+]+")
    text = re.sub(pun_eng, '', text)
    # 利用jieba进行分词
    #words = ' '.join(jieba.cut(text)).split(" ")
    #print(words)
    return text#' '.join(words)

def build_vocab(word_list):
    print('building vocab ...')
    vocab_dict = {}
    vocab_dict['PAD'] = 0
    vocab_dict['UNK'] = 1
    for w in word_list:
        vocab_dict[w] = vocab_dict.get(w,len(vocab_dict))
    return vocab_dict

if __name__ == "__main__":
    read_file('atec_nlp_sim_train.csv')
    #print(cut_words('我这个逾期后还完了最低还款后能分期吗'))