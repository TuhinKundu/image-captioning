import os
import pickle
from PIL import Image
from collections import Counter
import json
import numpy as np


def image_preprocess(folder_path, size):
    li_folder= folder_path.split('/')
    resized_folder = li_folder[-1]+'_resized_' + str(size)
    resized_folder_path = '/'.join(li_folder[:-1])+ '/' +resized_folder
    print(resized_folder_path)
    if not os.path.exists(resized_folder_path):
        os.makedirs(resized_folder_path)
    file_list = os.listdir(folder_path)
    print('No of images in your folder: '+str(len(file_list)))
    cnt=0
    for image in file_list:
        if cnt%100==0:
            print("Images processed: "+str(cnt))
        try:
            im = Image.open(folder_path + '/' + image)
        except:
            print("File not processed:"+image)
            continue
        im = im.resize([size, size], Image.ANTIALIAS)
        im.save(resized_folder_path + '/' + image)
        cnt+=1


def create_vocab(dataset, karpathy_json_path, min_word_freq, max_len=100):

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0


    with open('dumps/vocab_'+dataset+'.pkl', 'wb') as f:
        pickle.dump(word_map,f)

    print(dataset+" vocabulary pickle dumped!!")


def glove_embedding(filename):
    file_glove=open(filename)
    glove={}
    for line in file_glove:
        tmp=line.split()
        word=tmp[0]
        coefficient=np.asarray(tmp[1:], dtype='float')
        glove[word]=coefficient

    file_glove.close()
    return glove


def create_embed_vocab(embeddings, embed_path, vocab_path, embed_dim):

    glove = glove_embedding(embed_path)
    print("Embedding successfully loaded!!")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    weight_matrix = np.zeros((len(vocab), embed_dim))
    pos=0
    for word in list(vocab.keys()):

        try:
            weight_matrix[vocab[word]] = glove[word]

        except KeyError:
            weight_matrix[pos] = np.random.normal(scale=0.6, size=(embed_dim, ))
        pos+=1
    print("Decoder weight matrix generated!")
    pickle.dump(weight_matrix, open('dumps/'+embeddings+'_'+str(embed_dim)+'.pkl','wb'), protocol=2)


class loss_obj(object):
    def __init__(self):
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


