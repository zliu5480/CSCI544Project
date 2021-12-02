import json
import pdb
import numpy as np
import random
from nltk.corpus import stopwords

class average_meter(object):
    '''Computes and stores the average and current value
    '''
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_model(file_path = "./data/glove.6B.300d.txt"):
    glove = {}
    with open(file_path, "r", encoding='utf-8') as f:
        for lines in f:
            items = lines.split()
            if len(items) != 301:
                continue
            else:
                word_vector = []
                for i in range(1,301):
                    word_vector.append(float(items[i]))
                glove[items[0]] = word_vector
    UNK = "< UNK >"
    glove[UNK] = np.random.uniform(-0.25, 0.25, 300).tolist()
    return glove

def load_glove(path = 'GloVe.json'):
    with open('GloVe.json','r',encoding='utf8')as fp:
        json_data = json.load(fp)
    return json_data

def load_train(file_path = "./data/ISEAR.txt"):
    train_x = []
    train_y = []
    with open(file_path, "r", encoding='utf-8') as f:
        for lines in f:
            y_x = lines.split("|")
            train_y.append(y_x[0])
            train_x.append(y_x[1])
    return train_x, train_y

def load_dev(file_path = "./data/test.txt"):
    dev_x = []
    dev_y = []
    with open(file_path, "r", encoding='utf-8') as f:
        for lines in f:
            y_x = lines.split("|")
            dev_y.append(y_x[0])
            dev_x.append(y_x[2])
    return dev_x, dev_y

def split_train_test(data_X, data_Y, test_ratio):
    
    #combined_lsts = list(zip(data_X, data_Y))
    #random.shuffle(combined_lsts)
    test_set_size = int(len(data_X) * test_ratio)
    #data_X, data_Y = list(zip(*combined_lsts))
    train_x_raw = data_X[:test_set_size]
    train_y_raw = data_Y[:test_set_size]
    dev_x_raw = data_X[test_set_size:]
    dev_y_raw = data_Y[test_set_size:]

    return train_x_raw, train_y_raw, dev_x_raw, dev_y_raw


contractions_dict = {
    "i'm" : "i am",
    "i'll" : "i will",
    "i'd" : "i would",
    "i've" : "i have",
    "you're" : "you are",
    "you'll" : "you will",
    "you'd" : "you would",
    "you've" : "you have",
    "she's" : "she is",
    "she'll" : "she will",
    "he's" : "he is",
    "he'll" : "he will",
    "he'd" : "he would",
    "they're" : "they are",
    "they'll" : "they will",
    "they'd" : "they would",
    "that's" : "that is",
    "that'll" : "that will",
    "that'd" : "that would",
    "who's" : "who is",
    "who'll" : "who will",
    "who'd" : "who would",
    "what's" : "what is",
    "what're" : "what are",
    "what'll" : "what will",
    "what'd" : "what would",
    "where's" : "where is",
    "where'll" : "where will",
    "where'd" : "where would",
    "when's" : "when is",
    "when'll" : "when will",
    "when'd" : "when would",
    "why's" : "why is",
    "why'll" : "why will",
    "why'd" : "why would",
    "how's" : "how is",
    "how'll" : "how will",
    "how'd" : "how would",
    "would've" : "would have",
    "should've" : "should have",
    "could've" : "could have",
    "might've" : "might have",
    "must've" : "must have",
    "isn't" : "is not",
    "aren't" : "are not",
    "wasn't" : "was not",
    "weren't" : "were not",
    "haven't" : "have not",
    "hasn't" : "has not",
    "hadn't" : "had not",
    "won't" : "will not",
    "wouldn't" : "would not",
    "don't" : "do not",
    "doesn't" : "does not",
    "didn't" : "did not",
    "can't" : "cannot",
    "couldn't" : "could not",
    "shouldn't" : "should not",
    "mightn't" : "might not",
    "mustn't" : "must not"
}

def contractionfunction(s):
    if s in contractions_dict.keys():
        return contractions_dict[s]
    return s

def preprocess(sentence_list):
    stop = stopwords.words('english')
    char_replace = {",",".","/",";","'","[","]","\\","!","@","#","$","%","^","&","*","(",")","-","_","=","+","<",">","?",":","\"","{","}","|"}
    for i in range(len(sentence_list)):
        sentence_list[i] = sentence_list[i].lower()
        for char in char_replace:
            if char in sentence_list[i]:
                sentence_list[i] = sentence_list[i].replace(char, " ")
        sentence_list[i] = ' '.join([word for word in sentence_list[i].split() if word not in (stop)])
        sentence_list[i] = sentence_list[i].split()
    return sentence_list

def sort_key(a):
    return a[1]

def generate_label2id(y_data):
    y_count = {}
    label_list = []
    for y in y_data:
        if y in y_count:
            y_count[y] += 1
        else:
            y_count[y] = 1
    
    for key in y_count:
        item = y_count[key]
        label_list.append((key,item))
    label_list.sort(reverse=True,key=sort_key)

    label2id = {}
    i = 0
    for label in label_list:
        label2id[label[0]] = i
        label2id[i] = label[0]
        i+=1
    return label2id

def get_type_glove(unk):
    digits = 0
    for c in unk:
        if c.isdigit():
            digits += 1
    df = digits/len(unk)
    if unk.isdigit():
        return 1.0
    elif df > 0.5:
        return 2.0
    elif digits>0:
        return 3.0
    else:
        return 0.0





