from utilities import *
from Encoder import *
import json
images_path_name = 'dataset/flickr30k-images'

#image_preprocess(images_path_name,224)

#create_vocab('flickr8k', 'caption_datasets/dataset_flickr8k.json', 5)
#create_vocab('flickr30k', 'caption_datasets/dataset_flickr30k.json',5)
#create_vocab('coco',  'caption_datasets/dataset_coco.json', 5)


#create_embed_vocab('glove_twitter_27B','../Embeddings/glove.twitter.27B.200d.txt', 'dumps/vocab_flickr8k.pkl', 200)



with open('caption_datasets/dataset_flickr30k.json', 'rb') as f:
    data = json.load(f)
'''
for img in data['images']:
    captions = []   
    for c in img['sentences']:
        if len(c['tokens']) <= 100:
            captions.append(c['tokens'])
    for key in list(img.keys()):
        print(key+': '+str(img[key]))
    print(captions)
    print()
    

img_ids = []
img_map = {}
for img in data['images']:
    if img['split'] == 'train':
        img_ids.append(img['imgid'])
        img_map[img['imgid']] = {}
        img_map[img['imgid']]['filename'] = img['filename']
        captions = []
        for c in img['sentences']:
            captions.append(c['tokens'])
        img_map[img['imgid']]['caption'] = captions[-1]


for img in list(img_map.keys()):
    print(img)
    print(img_map[img])
    print()
'''
from data_loader import *
with open('dumps/vocab_flickr8k.pkl', 'rb') as f:
    vocab = pickle.load(f)
train_load= get_loader('val', 'flickr8k', vocab, 224, 32)
print(train_load)