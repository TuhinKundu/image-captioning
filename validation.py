from nltk.translate.bleu_score import corpus_bleu
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import imageio
import skimage.transform
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
import torch
from utilities import *
import torch.nn as nn
from tqdm import tqdm
from data_loader import *
from Encoder import *
from Decoder import *
import argparse

parser = argparse.ArgumentParser(description='Model Name')
parser.add_argument('--encoder', type=str, default="resnet101")
parser.add_argument('--decoder', type=str, default='baseline')
parser.add_argument('--dataset', type=str, default='flickr8k')
flags = parser.parse_args()
encoder_type = flags.encoder
decoder_type = flags.decoder
dataset = flags.dataset

bert_model = False
glove_model = True
if glove_model == True:
    glove_pickle_path = 'dumps/glove_twitter_27B_200.pkl'
    glove_vectors = pickle.load(open(glove_pickle_path, 'rb'))
    glove_vectors = torch.tensor(glove_vectors)
else:
    glove_vectors = None


PAD = 0
START = 1
END = 2
UNK = 3


batch_size= 32
#Load dataset vocabulary
img_size=224
dataset = 'flickr8k'
vocab = pickle.load(open('dumps/vocab_'+dataset+'.pkl', 'rb'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

val_loader = get_loader('val', dataset, vocab, img_size, batch_size)

criterion = nn.CrossEntropyLoss().to(device)

encoder = Encoder_Resnet101().to(device)
decoder = Decoder(vocab_size=len(vocab), use_glove=glove_model, use_bert=bert_model, glove_vectors=glove_vectors, vocab=vocab).to(device)
#decoder_optimizer = torch.optim.Adam(params=decoder.parameters(),lr=decoder_lr)

def print_sample(hypotheses, references, test_references, imgs, alphas, k, show_att, losses):
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0, 1, 0, 0))
    bleu_3 = corpus_bleu(references, hypotheses, weights=(0, 0, 1, 0))
    bleu_4 = corpus_bleu(references, hypotheses, weights=(0, 0, 0, 1))

    print("Validation loss: " + str(losses.avg))
    print("BLEU-1: " + str(bleu_1))
    print("BLEU-2: " + str(bleu_2))
    print("BLEU-3: " + str(bleu_3))
    print("BLEU-4: " + str(bleu_4))

    img_dim = 336  # 14*24
    idx2word = {vocab[key] : key for key in list(vocab.keys()) }
    hyp_sentence = []
    for word_idx in hypotheses[k]:
        hyp_sentence.append(idx2word[word_idx])

    ref_sentence = []
    for word_idx in test_references[k]:
        ref_sentence.append(idx2word[word_idx])

    print('Hypotheses: ' + " ".join(hyp_sentence))
    print('References: ' + " ".join(ref_sentence))

    img = imgs[0][k]
    imageio.imwrite('sample'+encoder_type+'_'+decoder_type+'_'+dataset+'.jpg', img)

    if show_att:
        image = Image.open('sample'+encoder_type+'_'+decoder_type+'_'+dataset+'.jpg')
        image = image.resize([img_dim, img_dim], Image.LANCZOS)
        for t in range(len(hyp_sentence)):

            plt.subplot(np.ceil(len(hyp_sentence) / 5.), 5, t + 1)

            plt.text(0, 1, '%s' % (hyp_sentence[t]), color='black', backgroundcolor='white', fontsize=12)
            plt.imshow(image)
            current_alpha = alphas[0][t, :].detach().numpy()
            alpha = skimage.transform.resize(current_alpha, [img_dim, img_dim])
            if t == 0:
                plt.imshow(alpha, alpha=0)
            else:
                plt.imshow(alpha, alpha=0.7)
            plt.axis('off')
    else:
        img = imageio.imread('sample'+encoder_type+'_'+decoder_type+'_'+dataset+'.jpg')
        plt.imshow(img)
        plt.axis('off')
        plt.show()


def validate():
    references = []
    test_references = []
    hypotheses = []
    all_imgs = []
    all_alphas = []

    encoder_checkpoint = torch.load('checkpoints/encoder_'+encoder_type+'_'+decoder_type+'.pt')
    decoder_checkpoint = torch.load('checkpoints/decoder_'+encoder_type+'_'+decoder_type+'.pt')

    encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    #decoder_optimizer = torch.optim.Adam(params=decoder.parameters(), lr=decoder_lr)
    decoder.load_state_dict(decoder_checkpoint['model_state_dict'])
    #decoder_optimizer.load_state_dict(decoder_checkpoint['optimizer_state_dict'])

    print("Started validation...")
    decoder.eval()
    encoder.eval()

    losses = loss_obj()

    num_batches = len(val_loader)
    # Batches
    for i, (imgs, caps, caplens) in enumerate(tqdm(val_loader)):

        imgs_jpg = imgs.numpy()
        imgs_jpg = np.swapaxes(np.swapaxes(imgs_jpg, 1, 3), 1, 2)

        # Forward prop.
        imgs = encoder(imgs.to(device))
        caps = caps.to(device)

        scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps, caplens)
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        # Calculate loss
        loss = criterion(scores_packed, targets_packed)
        loss += ((1. - alphas.sum(dim=1)) ** 2).mean()
        losses.update(loss.item(), sum(decode_lengths))

        # References
        for j in range(targets.shape[0]):
            img_caps = targets[j].tolist()  # validation dataset only has 1 unique caption per img
            clean_cap = [w for w in img_caps if w not in [PAD, START, END]]  # remove pad, start, and end
            img_captions = list(map(lambda c: clean_cap, img_caps))
            test_references.append(clean_cap)
            references.append(img_captions)

        # Hypotheses
        _, preds = torch.max(scores, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            pred = p[:decode_lengths[j]]
            pred = [w for w in pred if w not in [PAD, START, END]]
            temp_preds.append(pred)  # remove pads, start, and end
        preds = temp_preds
        hypotheses.extend(preds)

        if i == 0:
            all_alphas.append(alphas)
            all_imgs.append(imgs_jpg)

    print("Completed validation...")
    print_sample(hypotheses, references, test_references, all_imgs, all_alphas, 1, False, losses)


validate()