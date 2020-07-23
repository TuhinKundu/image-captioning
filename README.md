# Image Captioning

Repository containing code for course project done for CS515 Advanced Computer Vision at the University of Illinois at Chicago [report](https://github.com/TuhinKundu/image-captioning/blob/master/CS515_report.pdf).

Model contains encoder-decoder architecture where Resnet-101 is used as the encoder and various decoders using LSTM, Glove embeddings and BERT model are used to perform an empirical analysis.

### Requirements

* torch
* transformers

### Dataset and other requirements
* [Andrej Karpathy's training, validation and test splits](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)
* [COCO 2014 dataset](https://cocodataset.org/#download)
* [Flickr8k dataset](https://academictorrents.com/details/9dea07ba660a722ae1008c4c8afdd303b6f6e53b) (Original link is broken)
* [Flickr30k dataset](http://shannon.cs.illinois.edu/DenotationGraph/)
* [Glove embeddings](https://nlp.stanford.edu/projects/glove/) (Twitter, 200d, glove.twitter.27B.zip)

### Abstract

With an immense amount of visual information being
generated and aggregated from various sources, making
sense of this information and organising this data is becoming increasingly important. Image captioning is generating
a meaningful grammatically correct sentence to understand
the scene holistically. It translates visual information to
textual information by generating a description of an image using deep learning models. We use datasets such as
Flickr8, Flickr30 and COCO 2014 to investigate the tradeoff between using various combinations of encoder-decoder
based models which comprise of convolutional and recurrent neural networks. We add embeddings obtained from
language models such as Glove and BERT as weight initialization to our decoder unit to check for performance gains
and to lower training time. We conclude that contextual embeddings obtained from BERT provide a significant performance in terms of BLEU score due to taking into account
contextual information that may be present in captions or
sentences. 


### Credits

Repositories I referred to for the project:

* [https://github.com/ajamjoom/Image-Captions](https://github.com/ajamjoom/Image-Captions)
* [https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
