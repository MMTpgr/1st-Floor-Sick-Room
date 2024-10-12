import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self, index, targets):
        logits = self.token_embedding_table(index)
        B,T,C = logits.shape   #batch time ( sequence of integers) channels (vocab size)
        logits = logits.view(B*T,C)
        targets.view(B*T)
        loss = F.cross_entropy(logits,targets )

        return logits 
        

      
    