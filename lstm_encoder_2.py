import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Sample sentences
sentences = [
    ["I", "love", "cats"],
    ["I", "love", "dogs"],
    ["We", "love", "cats"],
    ["We", "all", "dogs"]
]

# Step 1: Create Vocabulary and Numericalize Sentences
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def build_vocab(self, sentences):
        for sentence in sentences:
            for word in sentence:
                self.add_word(word)
        # Add special tokens
        self.add_word('<PAD>')
        self.add_word('<UNK>')
        
    def numericalize(self, sentence):
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in sentence]

# Build vocabulary
vocab = Vocabulary()
vocab.build_vocab(sentences)
print("Vocabulary:")
print(vocab.word2idx)
print("\nNumericalized sentences:")
for sentence in sentences:
    print(f"{sentence} => {vocab.numericalize(sentence)}")

# Step 2: Create Dataset and DataLoader
class SentenceDataset(Dataset):
    def __init__(self, sentences, vocab):
        self.sentences = sentences
        self.vocab = vocab
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        numericalized = self.vocab.numericalize(sentence)
        return torch.tensor(numericalized, dtype=torch.long)

# Create dataset and dataloader
dataset = SentenceDataset(sentences, vocab)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# Step 3: Define LSTM Encoder
class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
    def forward(self, src):
        # src shape: [batch_size, seq_len]
        embedded = self.embedding(src)  # [batch_size, seq_len, embedding_dim]
        _, (hidden, _) = self.lstm(embedded)  # hidden: [1, batch_size, hidden_dim]
        return hidden.squeeze(0)  # Remove num_layers dimension

# Step 4: Initialize and Run Encoder
VOCAB_SIZE = len(vocab.word2idx)
EMBEDDING_DIM = 10
HIDDEN_DIM = 16

encoder = LSTMEncoder(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

print("\nEncoding results:")
for batch in dataloader:
    # batch shape: [4, 3] (batch_size, seq_len)
    encoded_output = encoder(batch)
    print(f"\nEncoded vectors shape: {encoded_output.shape}")  # Should be [4, 16]
    
    # Convert to numpy for readability
    encoded_np = encoded_output.detach().numpy()
    
    print("\nSentence encodings:")
    for i, sentence in enumerate(sentences):
        print(f"{' '.join(sentence):<15} => {np.round(encoded_np[i], 4)}")