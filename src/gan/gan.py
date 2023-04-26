import torch
from tqdm.notebook import tqdm as ntqdm
from torch.optim import Adam
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt

class GANS():
    def __init__(
        self,
        model,
        data_loader,
        sde,
        sampler,
        eps=1e-5 
    ) -> None:
        self.model = model
        self.sde = sde
        self.data_loader = data_loader
        self.sampler = sampler
        self.score_model = torch.nn.DataParallel(model(self.sde.marginal_prob)).to(device)
        self.rsde = self.sde.reverse(self.score_model)
        self.eps = eps

    def _loss_fn(self, x: torch.Tensor) -> float:
        random_t = torch.rand(x.shape[0], device=x.device) * (1. - self.eps) + self.eps
        z = torch.randn_like(x)
        mean, std = self.sde.marginal_prob(x, random_t)
        perturbed_data = mean + z * std[:, None, None, None]
        score = self.score_model(perturbed_data, random_t)
        loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
        return loss

    def train_model(self, n_epochs: int = 50, batch_size: int = 32, lr: float = 1e-4) -> None:
      optimizer = Adam(self.score_model.parameters(), lr=lr)
      # tqdm_epoch = tqdm.trange(n_epochs)
      self.losses = []

      for epoch in ntqdm(range(n_epochs)):
          avg_loss = 0
          num_items = 0
        
          for x, y in self.data_loader:
              x = x.to(device)
              loss = self._loss_fn(x)
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
              avg_loss += loss.item() * x.shape[0]
              num_items += x.shape[0]
          #tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
          self.losses.append(avg_loss / num_items) 
          
          torch.save(self.score_model.state_dict(), 'ckpt.pth')

      self.x_size = x.shape

    def load_model(self, path: str = 'ckpt.pth') -> None:
        ckpt = torch.load(path, map_location=device)
        self.score_model.load_state_dict(ckpt)

    def _plot_loss(self):
        abs = np.arange(len(self.losses))
        plt.plot(abs, self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

    def sampling(self, shape: int = 1) -> torch.Tensor:

        self.sampler.sde = self.sde
        self.sampler.score_model = self.score_model
        self.sampler.shape = [shape, self.x_size[1], self.x_size[2], self.x_size[3]]

        self.samples = self.sampler.sampling()

        return self.samples

    def direct_sampling(self, shape: int = 1) -> torch.Tensor:

        self.sampler.sde = self.sde
        self.sampler.score_model = self.score_model
        self.sampler.shape = [shape, self.sampler.shape[1], self.sampler.shape[2], self.sampler.shape[3]]

        self.samples = self.sampler.sampling()

        # return self.samples

    def plot_samples(self, grid: bool = True) -> None:
        
        if grid:
            sample_grid = make_grid(self.samples, nrow=int(np.sqrt(self.samples.shape[0])))

            plt.figure(figsize=(6,6))
            plt.axis('off')
            plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
            plt.show()
        
        else:
            for i in range(self.samples.shape[0]):
                plt.imshow(self.samples[i][0].to('cpu'))
                plt.show()