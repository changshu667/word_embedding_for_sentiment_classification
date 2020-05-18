"""

"""


# Built-in
import os

# Libs
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split

# Own modules
from tqdm import tqdm
from models import cnw, sswe
from dudu_utils import process_block, preprocess
from main_cnw import load_csv_data, data_split, integer_encoding, DATA_DIR, DS_LEN, BUFFER_DIR, WINDOW_SIZE, VAL_PER, \
    RANDOM_SEED, GRAM, BATCH_SIZE, EMBED_SIZE, FC_NUMS


def make_ds():
    # load data
    df = load_csv_data(os.path.join(DATA_DIR, 'training.1600000.processed.noemoticon.csv'),
                       columns=['target', 'id', 'date', 'flag', 'user', 'text'])
    lbl = (np.array(df['target']) // 4).astype(np.int)[:DS_LEN]
    ftr = list(df['text'])[:DS_LEN]
    assert len(ftr) == len(lbl)

    # text cleaning
    preprocess_pb = process_block.ProcessBlock(preprocess.make_preprocessor, BUFFER_DIR,
                                               process_name='make_preprocessor_v2')
    ftr = preprocess_pb.run(
        force_run=False,
        preprocess_names=['lower_text', 'remove_punctuation', 'tokenize', 'stemming'],
        val=ftr
    )

    # train test split
    ftr = data_split.create_gram(ftr, GRAM)  # create n-gram
    ftr, lut, vocab_size = integer_encoding(ftr)  # integer encode
    new_ftr, new_lbl = [], []
    for f, l in zip(ftr, lbl):
        if len(f) > WINDOW_SIZE:
            new_ftr.append(f)
            new_lbl.append(l)
    ftr, lbl = new_ftr, new_lbl
    assert len(ftr) == len(lbl)

    # train test split
    x_train, x_valid, y_trian, y_valid = train_test_split(ftr, lbl, test_size=VAL_PER, random_state=RANDOM_SEED,
                                                          shuffle=True)
    x_train, x_valid, y_trian, y_valid = train_test_split(x_valid, y_valid, test_size=0.5, random_state=RANDOM_SEED,
                                                          shuffle=True)

    # make dataset
    print('Vocabulary look up table size {}'.format(vocab_size))
    ds_train = sswe.SSWEHataLoader(x_train, y_trian, WINDOW_SIZE)
    ds_valid = sswe.SSWEHataLoader(x_valid, y_valid, WINDOW_SIZE)
    ds_train = DataLoader(ds_train, BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=ds_train.collate_fn)
    ds_valid = DataLoader(ds_valid, BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=ds_valid.collate_fn)
    print('Training dataset size {}, validation dataset size {}'.format(len(y_trian), len(y_valid)))
    return ds_train, ds_valid, vocab_size


def load_model(model_path, vocab_size):
    # initialize model
    if 'cnw' in model_path:
        model = cnw.CNW(vocab_size, EMBED_SIZE, FC_NUMS)
    elif 'ssweh' in model_path or 'sswer' in model_path:
        model = sswe.SSWEH(vocab_size, EMBED_SIZE, FC_NUMS)
    elif 'ssweu' in model_path:
        model = sswe.SSWEU(vocab_size, EMBED_SIZE, FC_NUMS)
    else:
        raise NotImplementedError

    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage)['state_dict'])
    return model


def infer_dataset(model, dataset, device):
    model.to(device)
    model.eval()
    ftrs = []
    lbls = []
    for data in tqdm(dataset, desc='Inferring dataset'):
        txt, lbl = data
        txt = txt.to(device)
        lbl = lbl.to(device)
        embed = model.embed(txt)

        ftrs.append(embed.detach().cpu().numpy())
        lbls.append(lbl.detach().cpu().numpy())
    return np.concatenate(ftrs, axis=0), np.concatenate(lbls, axis=0)


if __name__ == '__main__':
    model_dict = {
        'C&W': r'./models/cnw-2020-04-29_20-02-45.pth.tar',
        '$SSWE_h$': r'./models/ssweh-2020-04-29_15-44-54.pth.tar',
        '$SSWE_r$': r'./models/sswer-2020-04-29_15-47-38.pth.tar',
        '$SSWE_u$': r'./models/ssweu-2020-04-29_18-41-13.pth.tar',
    }
    plt.figure(figsize=(8, 6))
    lw = 2
    ds_train, ds_valid, vocab_size = make_ds()

    for model_name, model_path in model_dict.items():
        print('Evaluating model: {}'.format(model_name))
        # model_path = r'./models/sswer-2020-04-28_22-15-49.pth.tar'
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = load_model(model_path, vocab_size)

        # two fold cross validation
        preds, gts = [], []
        for cnt, (trainset, valset) in enumerate(zip([ds_train, ds_valid], [ds_valid, ds_train])):
            print('Running on fold {}...'.format(cnt))
            # infer sentiment features
            ftrs_train, lbls_train = infer_dataset(model, trainset, device)
            ftrs_valid, lbls_valid = infer_dataset(model, valset, device)

            # train sentiment classifier
            clf = LogisticRegression(max_iter=1000)
            clf.fit(ftrs_train, lbls_train)

            # eval sentiment classifier
            pred = clf.predict_proba(ftrs_valid)[:, 1]
            preds.append(pred)
            gts.append(lbls_valid)
        preds = np.concatenate(preds, axis=0)
        gts = np.concatenate(gts, axis=0)

        fpr, tpr, _ = roc_curve(gts, preds)
        roc_auc = auc(fpr, tpr)
        f1 = f1_score(gts, preds >= 0.5)

        plt.plot(fpr, tpr, lw=lw, label='{} (AUC={:.2f}, F1={:.2f})'.format(model_name, roc_auc, f1))
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Comparison')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
