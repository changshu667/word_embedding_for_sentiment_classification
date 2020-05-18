"""
This is the main file for the SSWE project
TODO:
1. Read data
2. Preprocess data
3. Build model
4. Train model
5. Evaluate model
"""


# Built-in
import os
import datetime

# Libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Custom
from models import sswe
from dudu_utils import data_loader, preprocess, process_block, data_split

# Settings
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TOP_WORD = 10
DS_LEN = 1600000              # 1600000
BUFFER_DIR = './buffer'
GRAM = 1
VAL_PER = 0.2
RANDOM_SEED = 1
WINDOW_SIZE = 3
EMBED_SIZE = 50
FC_NUMS = (20, )
LEARN_RATE = 1e-4
EPOCH = 20
BATCH_SIZE = 100
SAVE_DIR = r'./models'
ALPHA = 0.5
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Devices being used:', DEVICE)


def integer_encoding(data):
    text_list = set()
    for d in tqdm(data):
        text_list.update(d)
    int_lut = {a: i for i, a in enumerate(list(text_list))}

    tsfmed_data = []
    for d in tqdm(data):
        tsfm = [int_lut[a] for a in d]
        tsfmed_data.append(tsfm)

    return tsfmed_data, int_lut, len(text_list)


def load_csv_data(file_name, columns=None, encoding='ISO-8859-1', header=None):
    """
    Load csv data and name the columns if given
    :param file_name: path to the csv file
    :param columns: user defined column names, or None
    :param encoding: encoding method of the csv file
    :param header: what header to be used for the data frame
    :return: pandas dataframe
    """
    df = data_loader.load_file(file_name, encoding=encoding, header=header)
    if columns is not None:
        assert isinstance(columns, list)
        df.columns = columns
    return df


def create_sliding_window(ftr, lbl, window_size, step_size):
    new_ftr, new_lbl = [], []
    for f, l in zip(ftr, lbl):
        for i in range(0, len(f)-window_size, step_size):
            new_ftr.append(f[i:i+window_size])
            new_lbl.append(l)
    return np.array(new_ftr), np.array(new_lbl)


def main():
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
    ftr = data_split.create_gram(ftr, GRAM)         # create n-gram
    ftr, lut, vocab_size = integer_encoding(ftr)                # integer encode
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

    # make dataset
    print('Vocabulary look up table size {}'.format(vocab_size))
    ds_train = sswe.SSWEUataLoader(x_train, y_trian, vocab_size, WINDOW_SIZE)
    ds_valid = sswe.SSWEUataLoader(x_valid, y_valid, vocab_size, WINDOW_SIZE)
    print('Training dataset size {}, validation dataset size {}'.format(len(ds_train), len(ds_valid)))
    datasets = {'train': DataLoader(ds_train, BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=ds_train.collate_fn),
                'valid': DataLoader(ds_valid, BATCH_SIZE, shuffle=False, drop_last=True, collate_fn=ds_valid.collate_fn)}

    # create model
    model = sswe.SSWEU(vocab_size, EMBED_SIZE, FC_NUMS)
    model.to(DEVICE)
    print('Total params: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    # define the loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
    criterion_syntax = nn.HingeEmbeddingLoss().to(DEVICE)
    criterion_sentim = nn.HingeEmbeddingLoss().to(DEVICE)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60], gamma=0.1)

    # train the model
    loss_history = {'train': [], 'valid': []}
    acc_history = {'train': [], 'valid': []}
    for epoch in range(EPOCH):
        for phase in ['train', 'valid']:
            losses = []
            accuracies = []
            crpt_ind = Variable(torch.LongTensor(BATCH_SIZE).fill_(-1), requires_grad=False).to(DEVICE)
            for data in tqdm(datasets[phase], desc='Training on epoch {}...'.format(epoch)):
                if phase == 'train':
                    model.train()
                    model.zero_grad()
                else:
                    model.eval()
                orig, crpt, lbl = data
                orig = orig.to(DEVICE)
                crpt = crpt.to(DEVICE)
                lbl = lbl.to(DEVICE)
                syntax_orig, sentim_orig = model(orig)
                syntax_crpt, sentim_crpt = model(crpt)

                syntax_orig, sentim_orig = torch.mean(syntax_orig, dim=1), torch.mean(sentim_orig, dim=1)
                syntax_crpt, sentim_crpt = torch.mean(syntax_crpt, dim=1), torch.mean(sentim_crpt, dim=1)

                pred_soft = sentim_orig[:, 0]

                syntax_loss = criterion_syntax(syntax_orig-syntax_crpt, crpt_ind)
                sentim_loss = criterion_sentim((sentim_orig-sentim_crpt)*((lbl-0.5)*2), crpt_ind)

                loss = ALPHA * syntax_loss + (1 - ALPHA) * sentim_loss
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                losses.append(loss.detach().cpu().numpy())
                fpr, tpr, _ = metrics.roc_curve(lbl.detach().cpu().numpy(),
                                                pred_soft.detach().cpu().numpy())
                accuracies.append(metrics.auc(fpr, tpr))
            epoch_loss = np.mean(losses)
            epoch_acc = np.mean(accuracies)
            print('Epoch {} ({}): Loss: {:.3f} Auc: {:.2f}'.format(epoch, phase, epoch_loss, epoch_acc))
            loss_history[phase].append(epoch_loss)
            acc_history[phase].append(epoch_acc)
        scheduler.step()

    # save model
    curr_time = datetime.datetime.now()
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    name_str = 'ssweu-{}'.format(curr_time.strftime('%Y-%m-%d_%H-%M-%S'))
    save_name = os.path.join(SAVE_DIR, '{}.pth.tar'.format(name_str))
    torch.save({
        'state_dict': model.state_dict(),
        'opt_dict': optimizer.state_dict(),
        'losses': loss_history,
        'accuracies': acc_history,
        'lut': lut,
    }, save_name)
    print('Saved model at {}'.format(save_name))

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(np.arange(EPOCH), loss_history['train'], label='train', marker='o')
    plt.plot(np.arange(EPOCH), loss_history['valid'], label='valid', marker='o')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('$SSWE_u$ Loss')
    plt.legend(loc='upper right')

    plt.subplot(122)
    plt.plot(np.arange(EPOCH), acc_history['train'], label='train', marker='o')
    plt.plot(np.arange(EPOCH), acc_history['valid'], label='valid', marker='o')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.title('$SSWE_u$ Auc')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join('./figures', '{}.png'.format(name_str)))
    plt.close()


if __name__ == '__main__':
    main()
