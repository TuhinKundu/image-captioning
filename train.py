import pickle
import json
from data_loader import *
from transformers import BertModel, BertTokenizer
from utilities import *
from Encoder import *

from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
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


glove_model = False
bert_model = False
glove_vectors = None

if decoder_type=='glove':
    glove_model = True
    glove_pickle_path = 'dumps/glove_twitter_27B_200.pkl'
    glove_vectors = pickle.load(open(glove_pickle_path, 'rb'))
    glove_vectors = torch.tensor(glove_vectors)
elif decoder_type == 'bert':
    bert_model = True


batch_size= 32
#Load dataset vocabulary
img_size=224

vocab = pickle.load(open('dumps/vocab_'+dataset+'.pkl', 'rb'))
print(encoder_type, decoder_type, dataset)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
BertModel = BertModel.from_pretrained('bert-base-uncased').to(device)
BertModel.eval()





# Model hyperparameters
grad_clip = 5.
num_epochs = 10
batch_size = 32
decoder_lr = 0.0004

# if both are false them model = baseline





PAD = 0
START = 1
END = 2
UNK = 3

train_loader = get_loader('train', dataset, vocab, img_size, batch_size)
#train_loader = get_loader('train', vocab, batch_size)



criterion = nn.CrossEntropyLoss().to(device)

encoder = Encoder_Resnet101().to(device)
decoder = Decoder(vocab_size=len(vocab),use_glove=glove_model, use_bert=bert_model, glove_vectors=glove_vectors, vocab=vocab).to(device)
decoder_optimizer = torch.optim.Adam(params=decoder.parameters(),lr=decoder_lr)


def train():
    print("Started training...")
    epoch=0
    loss=-1
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
        }, 'checkpoints/decoder_epoch' + str(epoch + 1)+'.pt')

        torch.save({
            'epoch': epoch,
            'model_state_dict': encoder.state_dict(),
            'loss': loss,
        }, 'checkpoints/encoder_epoch' + str(epoch + 1)+'.pt')

        print('epoch checkpoint saved')

    torch.save({
        'epoch': epoch,
        'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': decoder_optimizer.state_dict(),
        'loss': loss,
    }, 'checkpoints/decoder_'+encoder_type+'_'+decoder_type+'.pt')

    torch.save({
        'epoch': epoch,
        'model_state_dict': encoder.state_dict(),
        'loss': loss,
    }, 'checkpoints/encoder_'+encoder_type+'_'+decoder_type+'.pt')

    print('epoch checkpoint saved')
    print("Completed training...")


train()