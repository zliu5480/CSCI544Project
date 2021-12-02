import json
import numpy as np


def load_model(file_path = "./data/glove.twitter.27B.200d.txt"):
    glove = {}
    with open(file_path, "r", encoding='utf-8') as f:
        for lines in f:
            items = lines.split()
            if len(items) != 201:
                continue
            else:
                word_vector = []
                for i in range(1,201):
                    word_vector.append(float(items[i]))
                glove[items[0]] = word_vector
    return glove
GloVe = load_model()

UNK = "< UNK >"
GloVe[UNK] = np.random.uniform(-0.25, 0.25, 200).tolist()

json_str = json.dumps(GloVe)
with open('GloVe.json', 'w') as json_file:
    json_file.write(json_str)