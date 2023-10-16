## !wget https://raw.githubusercontent.com/karpathy/char-nn/master/data/tinyshakespeare/input.txt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import math

torch.manual_seed(372)

## Hyperparameters
BATCH_SIZE = 32         ## Number of sequence that will process in parallel
BLOCK_SIZE = 256        ## Maximum context Length for prediction
MAX_ITER = 10000
EVAL_INTERVAL = 500
LEARNING_RATE = 5e-3
EVAL_ITERS = 100
N_EMBED = 256
N_HEAD = 8
N_LAYER = 8
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#____________________________________________________________________________________________________________


with open('Dataset/input.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# Tokenize input text: CHaracterize level (subject to change for better results)
# create a mapping for character to integer, and create a encpder and a decoder


stoi = {ch: i for i, ch, in enumerate(chars)}
itos = {i: ch for i, ch, in enumerate(chars)}

encode = lambda s: [stoi[ch] for ch in s]
# encoder: Take a string input, output a list of integer
decode = lambda l: ''.join([itos[ch] for ch in l] )
# decoder: Takes a list of integer as input and outputs a string


# Split the text into training and validation set
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

# Load the data
def get_batch(split):
    # Generate a small batch of data of inputs X and targets Y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])

    return x,y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss =  model(X,Y)
            losses[k] = loss.item()
        
        out[split] = losses.mean()
    model.train()
    return out
        


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*[Block(N_EMBED, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBED) ## Final layer Norm
        self.lm_head = nn.Linear(N_EMBED, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        
        B,T = idx.shape

        # idx and token are both (B,T) tensor of integers
        token_embedding = self.token_embedding_table(idx)  ##(B,T,C)    
        positional_embedding = self.positional_embedding_table(torch.arrange(T, device=device)) ##(B,T,C)
        x = token_embedding + positional_embedding ##(B,T,C)
        x = self.blocks(x)  ##(B,T,C)
        x = self.ln_f(x)  ##(B,T,C)

        logits = self.lm_head(x) ##(B,T,vocab_size)

        ## Loss Function
        '''
            As cross entropy function require the paramenter in multi-dimensional input
            Channels is require to be 2nd Parapeter .
            Therefore we need to reshape the logits and target variable accordingly
        '''
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -BLOCK_SIZE:]
            # Get the predictions
            logits, loss = self(idx_cond)

            # Focus only on last step --> logits becomes (B,C)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1) # Apply softmax to get probabilites -- (B,C)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) ## (B,1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx
    
model = GPTLanguageModel()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for iter in range(MAX_ITER):

    ## Evaluate the loss on train and val set, every once in a while
    if iter % EVAL_INTERVAL == 0 or iter == MAX_ITER-1:
        losses = estimate_loss()
        print(f"step{iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    ## Sample a batch of data
    xb, yb = get_batch('train')


    ##  Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

## Generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))