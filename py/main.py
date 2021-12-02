import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from utils import *
from emo_dataset import *

def data_process(data, label2id, GloVe):
    b_size = len(data)
    x_tensor = torch.zeros(b_size, 80, 201)
    y_tensor = torch.zeros(b_size).long()
    
    for b_index in range(b_size):
        x = data[b_index][0]
        y = data[b_index][1]
        for xy_index in range(len(x)):
            word = x[xy_index]
            if word in GloVe:
                x_vector = deepcopy(GloVe[word])
                one_more = get_type_glove(word)
                x_vector.append(one_more)
                _x = torch.FloatTensor(x_vector)
                x_tensor[b_index][xy_index] = _x
            else:
                unk_vector = deepcopy(GloVe["< UNK >"])
                one_more = get_type_glove(word)
                unk_vector.append(one_more)
                _unk = torch.FloatTensor(unk_vector)
                x_tensor[b_index][xy_index] = _unk
        y_tensor[b_index] = label2id[y]
    data_len = []
    for xy in data:
        data_len.append(len(xy[0]))
    return x_tensor, y_tensor, data_len

def main():

    n_epochs = 20
    lr = 0.001
    TRAIN_BATCH_SIZE = 32
    DEV_BATCH_SIZE = 96
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    print('loading Glove dictionary')
    Glove = load_glove()

    print('loading train and test data')
    #data_x, data_y = load_train()
    #train_x_raw, train_y_raw, dev_x_raw, dev_y_raw = split_train_test(data_x, data_y, 0.2)
    train_x_raw, train_y_raw = load_train(file_path = "./data/train.txt")
    dev_x_raw, dev_y_raw = load_train(file_path = "./data/test.txt")
    train_x_raw = preprocess(train_x_raw)
    dev_x_raw = preprocess(dev_x_raw)

    label2id = generate_label2id(train_y_raw)

    train_y = []
    dev_y = []
    for label in train_y_raw:
        train_y.append(label2id[label])
    for label in dev_y_raw:
        dev_y.append(label2id[label])

    print('preparing Dataset')
    train_dataset = emotion_dataset(train_x_raw,train_y_raw)
    dev_dataset = emotion_dataset(dev_x_raw,dev_y_raw)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn= lambda x: data_process(x, label2id, Glove))
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=DEV_BATCH_SIZE, shuffle=False, collate_fn= lambda x: data_process(x, label2id, Glove))

    print('preparing Model')
    model = CNN_Text().to(DEVICE)
    print('setingt optimization method')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    acc_model = None

    for epoch in range(n_epochs):
        torch.cuda.empty_cache()
        train_loss = train_cnn(model, train_dataloader, criterion, optimizer, epoch, DEVICE)
        torch.cuda.empty_cache()
        scheduler.step()
        valid_loss, valid_acc = validate_cnn(model, dev_dataloader, criterion, optimizer, epoch, DEVICE)
        if valid_acc > best_acc:
            acc_model = deepcopy(model)
            best_acc = valid_acc
        print("{}%, Epoch {}, train loss {:.8f}, validate loss {:.8f}, acc {:.8f}".format(100*(epoch+1)/n_epochs,str(epoch).zfill(2),train_loss,valid_loss,valid_acc))

    print('Best_val_acc:',acc_model)
    torch.save(acc_model, "blstm1.pt")

class CNN_Text(nn.Module):
    
    def __init__(self):
        super(CNN_Text, self).__init__()
        filter_sizes = [1,2,3,5]
        num_filters = 36
        n_classes = 7
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, 201)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(filter_sizes)*num_filters, n_classes)


    def forward(self, x):
        x = x.unsqueeze(1)  
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
        x = torch.cat(x, 1)
        x = self.dropout(x)  
        logit = self.fc1(x) 
        return logit

def train_cnn(model, train_loader, criterion, optimizer, epoch, DEVICE):
    train_loss = average_meter()
    model.train()
    for i in train_loader:
        x = i[0].to(DEVICE)
        y = i[1].to(DEVICE)
        l = i[2]
        pred = model.forward(x)
        optimizer.zero_grad()
        loss = criterion(pred, y).to(DEVICE)
        train_loss.update(loss.item(),x.size(0))
        loss.backward()
        optimizer.step()
    return train_loss.avg
        
def validate_cnn(model, dev_loader, criterion, optimizer, epoch, DEVICE):
    valid_loss = average_meter()
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        step = 0
        for i in dev_loader:
            x = i[0].to(DEVICE)
            y = i[1].to(DEVICE)
            l = i[2]
            pred = model.forward(x)
            loss = criterion(pred, y).to(DEVICE)
            valid_loss.update(loss.item(),x.size(0))
            pred = torch.max(pred, 1)[1]
            correct += (pred == y).float().sum()
            total += y.shape[0]
            step += 1
    return valid_loss.avg, correct/total


if __name__ == "__main__":
    main()