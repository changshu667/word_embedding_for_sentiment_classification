"""

"""


# Built-in
import os

# Libs
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression

# Own modules
from infer_sentence import make_ds
from dudu_utils import preprocess, process_block
from inference import load_model, infer_dataset

# Settings
# model_path = r'./models/cnw-2020-04-29_20-02-45.pth.tar'
model_path = r'./models/sswer-2020-04-29_15-47-38.pth.tar'
checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
lut = checkpoint['lut']
model = load_model(model_path, len(lut))
window_size = 3
step_size = 3
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BUFFER_DIR = './buffer'


def train_clf(network, dev):
    dataset, vocab_size = make_ds()
    ftrs, lbls = infer_dataset(network, dataset, dev)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(ftrs, lbls)
    return clf


def pred_sentence():
    # input_ = r"this is the best class i have ever taken"
    # input_ = r'my results are soooo bad and not making any sense :('
    input_ = r"and i really don't like staying at home :("
    print('Input string: {}'.format(input_))

    # pre-processing
    input_ = preprocess.make_preprocessor(['lower_text', 'remove_punctuation', 'tokenize', 'stemming'], [input_])[0]
    print('After preprocessing: ', input_)

    # encoding
    input_ = [lut[a] for a in input_ if a in lut.keys()]
    print('After encoding: ', input_)

    # inference
    windows = []
    for i in range(0, len(input_)-window_size+1, step_size):
        windows.append(input_[i:i+window_size])
    if len(input_) % 3 != 0:
        windows.append(input_[-3:])
    print('Input to neural network: {}'.format(windows))
    input_ = torch.unsqueeze(torch.from_numpy(np.array(windows)), dim=0)
    embed = model.embed(input_).detach().cpu().numpy()

    # predict
    pb = process_block.ProcessBlock(train_clf, BUFFER_DIR,
                                     process_name=os.path.splitext(os.path.basename(model_path))[0])
    clf = pb.run(False, network=model, dev=device)
    prob = clf.predict_proba(embed)
    print('Confidence of positive: {:.2f}'.format(prob[0, 1]))


def word_dist(word1, word2):
    # pre-processing
    w1 = preprocess.make_preprocessor(['lower_text', 'remove_punctuation', 'tokenize', 'stemming'], [word1])[0]
    w2 = preprocess.make_preprocessor(['lower_text', 'remove_punctuation', 'tokenize', 'stemming'], [word2])[0]

    # encoding
    w1, w2 = lut[w1[0]], lut[w2[0]]

    # inference
    w1 = torch.unsqueeze(torch.from_numpy(np.array(w1)), dim=0)
    w1 = model.embed_word(w1).detach().cpu().numpy()[0]
    w2 = torch.unsqueeze(torch.from_numpy(np.array(w2)), dim=0)
    w2 = model.embed_word(w2).detach().cpu().numpy()[0]

    print('{} to {} distance: {:.2f}'.format(word1, word2, np.linalg.norm(w1-w2)))


if __name__ == '__main__':
    # word_dist('good', 'bad')
    # word_dist('good', 'well')
    pred_sentence()
