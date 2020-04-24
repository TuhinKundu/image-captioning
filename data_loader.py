import torch
import torch.utils.data as data
from pycocotools.coco import COCO
import nltk
from PIL import Image
import os
from torchvision import transforms
import json


class DataLoader(data.Dataset):
    def __init__(self, root, img_ids, img_map, vocab, transform=None):

        self.root = root
        self.data = img_map
        self.ids = img_ids
        self.vocab = vocab
        self.transform = transform


    def __getitem__(self, index):

        vocab = self.vocab
        ann_id = self.ids[index]
        caption = self.data[ann_id]['caption']
        print(caption)
        #img_id = coco.anns[ann_id]['image_id']
        #path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(self.root+self.data[ann_id]['filename']).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)


        #tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        tokens = caption
        caption = []
        caption.append(vocab['<start>'])
        #caption.extend([vocab[token] for token in tokens])
        for token in tokens:
            try:
                caption.append(vocab[token])
            except KeyError:
                caption.append(vocab['<pad>'])

        caption.append(vocab['<end>'])
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    data.sort(key=lambda  x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths

def get_loader(method, dataset, vocab, img_size, batch_size=32):

    # train/validation paths

    root = 'dataset/'+dataset+ '_images_resized_'+str(img_size)+'/'

    with open('caption_datasets/dataset_'+dataset+'.json', 'rb') as file:
        data = json.load(file)

    img_ids=[]
    img_map={}

    for img in data['images']:
        if img['split']==method:
            img_ids.append(img['imgid'])
            img_map[img['imgid']] = {}
            img_map[img['imgid']]['filename'] = img['filename']
            captions = []
            for c in img['sentences']:
                captions.append(c['tokens'])
            img_map[img['imgid']]['caption'] = captions[-1]


    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])



    coco = DataLoader(root=root, img_ids=img_ids, img_map=img_map, vocab=vocab, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=1,
                                              collate_fn=collate_fn)
    return data_loader
