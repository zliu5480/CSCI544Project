import torch.utils.data as datasets


class emotion_dataset(datasets.Dataset):
    
    def __init__(self, word_lists, label_lists):
        self.word_lists = word_lists
        self.label_lists = label_lists

    def __getitem__(self, index):
        return self.word_lists[index], self.label_lists[index]
    
    def __len__(self):
        return len(self.word_lists)

