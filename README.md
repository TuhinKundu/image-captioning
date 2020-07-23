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


### Credits

Repositories I referred to for the project:

* [https://github.com/ajamjoom/Image-Captions](https://github.com/ajamjoom/Image-Captions)
* [https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
