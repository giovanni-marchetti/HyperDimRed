import torch

class EmbedOptim():
    #An abstract class for optimizers for embedding methods

    def __init__(self, embedder, lr):
        #embedder: Embedder , lr: Float
        
        self.embedder = embedder
        self.lr = lr
    
    def zero_grad(self):
        self.embedder.embeddings.grad = torch.zeros_like(self.embedder.embeddings) 
    
    def step(self):
        pass


class StandardOptim(EmbedOptim):
    def step(self,idx):
        with torch.no_grad():
            embeddings = self.embedder.embeddings
            # print(embeddings.grad)
            embeddings[idx] -= self.lr * self.embedder.embeddings.grad[idx]


class PoincareOptim(EmbedOptim):
    def step(self):
        with torch.no_grad():
            embeddings = self.embedder.embeddings
            norms = (embeddings**2).sum(-1).unsqueeze(-1)
            embeddings -= self.lr * (1 - norms)**2 * embeddings.grad / 4



