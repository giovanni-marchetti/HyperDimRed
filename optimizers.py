import torch

class EmbedOptim():
    #An abstract class for optimizers for embedding methods

    def __init__(self, embedder, lr):
        self.embedder = embedder
        self.lr = lr

    def zero_grad(self):
        self.embedder.embeddings.grad = torch.zeros_like(self.embedder.embeddings) 
    
    def step(self):
        pass


class StandardOptim(EmbedOptim):
    def step(self):
        with torch.no_grad():
            self.embedder.embeddings -= self.lr * self.embedder.embeddings.grad


class PoincareOptim(EmbedOptim):
    def step(self):
        with torch.no_grad():   
            embeddings = self.embedder.embeddings
            norms = (embeddings**2).sum(-1).unsqueeze(-1)
            embeddings -= self.lr * (1 - norms)**2 * embeddings.grad / 4
            clipped_norms = torch.clamp(torch.sqrt(norms), min=1. - .0001)
            embeddings.div(clipped_norms.unsqueeze(-1)) 
