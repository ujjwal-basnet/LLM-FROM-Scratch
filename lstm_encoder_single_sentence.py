import torch 
import torch.nn as nn 

class Encoder(nn.Module):
    def __init__(self, vocab_size ,embed_size , hidden_size):
        super().__init__()
        
        self.embedding= nn.Embedding(num_embeddings= vocab_size , embedding_dim= embed_size)
        self.lstm= nn.LSTM(input_size= embed_size ,hidden_size = hidden_size , batch_first= True ) ## input size should be = embeded size
        
        
    def forward(self, input_seq): 
        ## input sequence means token of the sentence like , i am ujjwal  , token might be , [0 ,1 2 ]
        embedded= self.embedding(input_seq)
        _ , (hidden, cell) = self.lstm(embedded) ## 
        return hidden ,cell 
    
    
    
    
        
sentence = "My name is ujjwal".split()
print("#"*75)
print(f"Sentence is {sentence}") # Sentence is ['My', 'name', 'is', 'ujjwal']

vocab= {word : id for id , word in enumerate(set(sentence))} ## output vocab is  {'is': 0, 'name': 1, 'ujjwal': 2, 'My': 3}
print(f"vocab is  {vocab}")

token_ids = [vocab[word] for word in sentence] #Token id  is  [3, 1, 0, 2]
print(f"Token ids  is  {token_ids}")

## convert into tensor
token_ids= torch.tensor(token_ids)
print("#"*75)
print() 

########################################################################################################

######## prepearing input for encoder ########## 
vocab_size= len(vocab)
embed_size=  8 
hidden_state = 4 

################### encoder ###########
encoder= Encoder(vocab_size= vocab_size , embed_size= embed_size , hidden_size= hidden_state)

########## forward call ############## 
hidden ,cell = encoder(token_ids)

################################################################################################################

print()
print("+"*70)
print(f" encoded context :  {hidden}")





        
        
        