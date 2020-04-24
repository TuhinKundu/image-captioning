import pickle
import json
from data_loader import *
from transformers import BertModel, BertTokenizer
from utilities import *
from Encoder import *
from Decoder import *
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

batch_size= 32
#Load dataset vocabulary
img_size=224
dataset = 'flickr8k'
vocab = pickle.load(open('dumps/vocab_'+dataset+'.pkl', 'rb'))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
BertModel = BertModel.from_pretrained('bert-base-uncased').to(device)
BertModel.eval()

# Load GloVe
glove_pickle_path = 'dumps/glove_twitter_27B_200.pkl'
glove_vectors = pickle.load(open(glove_pickle_path, 'rb'))
glove_vectors = torch.tensor(glove_vectors)


# Model hyperparameters
grad_clip = 5.
num_epochs = 4
batch_size = 32
decoder_lr = 0.0004

# if both are false them model = baseline

glove_model = False
bert_model = False

from_checkpoint = False
train_model = False
valid_model = True

PAD = 0
START = 1
END = 2
UNK = 3

train_loader = get_loader('train', dataset, vocab, img_size, batch_size)
#train_loader = get_loader('train', vocab, batch_size)
val_loader = get_loader('val', dataset, vocab, img_size, batch_size)


criterion = nn.CrossEntropyLoss().to(device)

if from_checkpoint:

    encoder = Encoder_Resnet101().to(device)
    decoder = Decoder(vocab_size=len(vocab),use_glove=glove_model, use_bert=bert_model).to(device)

    if torch.cuda.is_available():
        if bert_model:
            print('Pre-Trained BERT Model')
            encoder_checkpoint = torch.load('checkpoints/encoder_bert')
            decoder_checkpoint = torch.load('checkpoints/decoder_bert')
        elif glove_model:
            print('Pre-Trained GloVe Model')
            encoder_checkpoint = torch.load('checkpoints/encoder_glove')
            decoder_checkpoint = torch.load('checkpoints/decoder_glove')
        else:
            print('Pre-Trained Baseline Model')
            encoder_checkpoint = torch.load('checkpoints/encoder_baseline')
            decoder_checkpoint = torch.load('checkpoints/decoder_baseline')
    else:
        if bert_model:
            print('Pre-Trained BERT Model')
            encoder_checkpoint = torch.load('checkpoints/encoder_bert', map_location='cpu')
            decoder_checkpoint = torch.load('checkpoints/decoder_bert', map_location='cpu')
        elif glove_model:
            print('Pre-Trained GloVe Model')
            encoder_checkpoint = torch.load('checkpoints/encoder_glove', map_location='cpu')
            decoder_checkpoint = torch.load('checkpoints/decoder_glove', map_location='cpu')
        else:
            print('Pre-Trained Baseline Model')
            encoder_checkpoint = torch.load('checkpoints/encoder_baseline', map_location='cpu')
            decoder_checkpoint = torch.load('checkpoints/decoder_baseline', map_location='cpu')

    encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    decoder_optimizer = torch.optim.Adam(params=decoder.parameters(), lr=decoder_lr)
    decoder.load_state_dict(decoder_checkpoint['model_state_dict'])
    decoder_optimizer.load_state_dict(decoder_checkpoint['optimizer_state_dict'])
else:
    encoder = Encoder_Resnet101().to(device)
    decoder = Decoder(vocab_size=len(vocab),use_glove=glove_model, use_bert=bert_model, glove_vectors=glove_vectors).to(device)
    decoder_optimizer = torch.optim.Adam(params=decoder.parameters(),lr=decoder_lr)


def train():
    print("Started training...")
    for epoch in tqdm(range(num_epochs)):
        decoder.train()
        encoder.train()

        losses = loss_obj()
        num_batches = len(train_loader)

        for i, (imgs, caps, caplens) in enumerate(tqdm(train_loader)):

            imgs = encoder(imgs.to(device))
            caps = caps.to(device)

            scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps, caplens)
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]

            targets = caps_sorted[:, 1:]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

            loss = criterion(scores, targets).to(device)

            loss += ((1. - alphas.sum(dim=1)) ** 2).mean()

            decoder_optimizer.zero_grad()
            loss.backward()

            # grad_clip decoder
            for group in decoder_optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)

            decoder_optimizer.step()

            losses.update(loss.item(), sum(decode_lengths))

            # save model each 100 batches
            if i % 5000 == 0 and i != 0:
                print('epoch ' + str(epoch + 1) + '/4 ,Batch ' + str(i) + '/' + str(num_batches) + ' loss:' + str(
                    losses.avg))

                # adjust learning rate (create condition for this)
                for param_group in decoder_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.8

                print('saving model...')

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': decoder_optimizer.state_dict(),
                    'loss': loss,
                }, 'checkpoints/decoder_mid')

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': encoder.state_dict(),
                    'loss': loss,
                }, 'checkpoints/encode_mid')

                print('model saved')

        torch.save({
            'epoch': epoch,
            'model_state_dict': decoder.state_dict(),
            'optimizer_state_dict': decoder_optimizer.state_dict(),
            'loss': loss,
        }, 'checkpoints/decoder_epoch' + str(epoch + 1))

        torch.save({
            'epoch': epoch,
            'model_state_dict': encoder.state_dict(),
            'loss': loss,
        }, 'checkpoints/encoder_epoch' + str(epoch + 1))

        print('epoch checkpoint saved')

    print("Completed training...")


train()