import torch


class EmbedOptim():
    # An abstract class for optimizers for embedding methods

    def __init__(self, embedder, lr):
        # embedder: Embedder , lr: Float

        self.embedder = embedder
        self.lr = lr

    def zero_grad(self):
        self.embedder.embeddings.grad = torch.zeros_like(self.embedder.embeddings)

    def step(self):
        pass


class StandardOptim(EmbedOptim):
    def step(self, idx):
        with torch.no_grad():
            embeddings = self.embedder.embeddings
            # print(embeddings.grad)
            # embeddings[idx] -= self.lr * self.embedder.embeddings.grad[idx]

            grad = self.embedder.embeddings.grad.detach()
            embeddings[idx] -= self.lr * grad[idx]


class AdamOptim(EmbedOptim):
    def __init__(self, embedder, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(embedder,lr)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize moment estimates for each parameter
        self.m = torch.zeros_like(embedder.embeddings)
        self.v = torch.zeros_like(embedder.embeddings)
        self.t = 0

    def step(self, idx):
        with torch.no_grad():
            # Increment time step
            self.t += 1

            # Get the gradient of the embeddings
            grad = self.embedder.embeddings.grad.detach()

            # Compute biased first moment estimate
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad[idx]

            # Compute biased second moment estimate
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (grad[idx] ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second moment estimate
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

            # Update the embeddings using the Adam rule
            self.embedder.embeddings[idx] -= self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)

class PoincareOptim(EmbedOptim):
    def step(self, idx):
        with torch.no_grad():
            embeddings = self.embedder.embeddings
            norms = (embeddings ** 2).sum(-1).unsqueeze(-1)
            embeddings[idx] -= self.lr * (1 - norms[idx]) ** 2 * embeddings.grad[idx] / 4