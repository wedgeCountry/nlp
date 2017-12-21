import torch
import torch.nn as nn
from torch.autograd import Variable

import unidecode
import string
import random
import re

"""
Starting from 
https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb
"""

class CharTensor(Variable):

    all_characters = string.printable    
    n_characters = len(all_characters)
    
    def __new__(self, string):
        """
        Convert a string to a tensor in the char space. Each entry contains the index of the dimension in the char space for each char.
        """
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = self.all_characters.index(string[c])
        return Variable(tensor)

if __name__ == "__main__":
    print(string.printable)
    print(CharTensor('abcDEF'))


class TextDataSet(object):
    
    def __init__(self, src):
        """
        A text data set. Source is a text file.
        """
        super().__init__()
        with open(src, 'r') as src_file:
            self.text = unidecode.unidecode(src_file.read())
        self.length = len(self.text)
        
    def random_chunk(self, chunk_len = 200):
        """
        Generate a random subtext of specified length.
        """
        start_index = random.randint(0, self.length - chunk_len)
        end_index = start_index + chunk_len + 1
        return self.text[start_index:end_index]

    def random_data_point(self, chunk_len = 200):
        """
        Create a random data point. The first len-1 chars are the input, the last len-1 chars are the target.
        """
        chunk = self.random_chunk(chunk_len)
        inp = CharTensor(chunk[:-1])
        target = CharTensor(chunk[1:])
        return inp, target

if __name__ == "__main__":
    data_set = TextDataSet("data/Alice_in_Wonderland.txt")
    print(data_set.random_chunk(40))



class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        """
        Recurrent neural network to learn to write like a certain writer.
        """
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
    
    def train(self, inp, target, criterion):
        hidden = self.init_hidden()
        self.zero_grad()
        loss = 0
        for c in range(len(target)):
            output, hidden = self(inp[c], hidden)
            loss += criterion(output, target[c])

        loss.backward()
        decoder_optimizer.step()

        return loss.data[0] * 1./ len(target)

    def evaluate(self, prime_str='A', predict_len=100, temperature=0.8):
        hidden = self.init_hidden()
        prime_input = CharTensor(prime_str)
        predicted = prime_str

        # Use priming string to "build up" hidden state
        for p in range(len(prime_str) - 1):
            _, hidden = self(prime_input[p], hidden)
        inp = prime_input[-1]
        
        for p in range(predict_len):
            output, hidden = self(inp, hidden)
            
            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            
            # Add predicted character to string and use as next input
            predicted_char = CharTensor.all_characters[top_i]
            predicted += predicted_char
            inp = CharTensor(predicted_char)

        return predicted
    

n_epochs = 6000
print_every = 200
plot_every = 200
hidden_size = 100
n_layers = 1
lr = 0.005
n_chars = CharTensor.n_characters
chunk_len = 200

decoder = RNN(n_chars, hidden_size, n_chars, n_layers)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

import time, math
start = time.time()
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

data_set = TextDataSet("data/Alice_in_Wonderland.txt")

all_losses = []
loss_avg = 0

for epoch in range(1, n_epochs + 1):
    loss = decoder.train(*data_set.random_data_point(chunk_len), criterion)       
    loss_avg += loss

    if epoch % plot_every == 0:
        all_losses.append(loss_avg * 1.0 / plot_every)
        loss_avg = 0

    if epoch % print_every == 0:
        
        print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch * 100. / n_epochs, loss))
        print("Avg. loss: %s" % all_losses[-1])
        print(decoder.evaluate('Wh', 100), '\n')
        
