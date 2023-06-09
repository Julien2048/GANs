import torch
from tqdm.notebook import tqdm as ntqdm
from torch.optim import Adam
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, Food101
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from gan.utils import saved_model_paths
from gan.sde import find_sde


class GANS:
    def __init__(
        self,
        model,
        data_loader,
        sde,
        sampler,
        eps=1e-5,
        device: str = "cuda",
        batch_size: int = 128,
        size: int = 128,
    ) -> None:
        self.model = model
        self.sde = sde
        self.data_loader = data_loader
        self.sampler = sampler
        self.device = device
        self.score_model = torch.nn.DataParallel(model(self.sde.marginal_prob_std)).to(
            self.device
        )
        self.rsde = self.sde.reverse(self.score_model)
        self.eps = eps
        self.batch_size = batch_size
        self.size = size
        self.sde.device = self.device

        # self._get_size_ex()
        self._load_dataset()

    def _load_dataset(self) -> None:
        if self.data_loader == "MNIST":
            self.data_loader_str = self.data_loader
            dataset = MNIST(
                ".", train=True, transform=transforms.ToTensor(), download=True
            )
            self.data_loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
            )
        if self.data_loader == "FMNIST":
            self.data_loader_str = self.data_loader
            dataset = FashionMNIST(
                ".", train=True, transform=transforms.ToTensor(), download=True
            )
            self.data_loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
            )

        if self.data_loader == "Food101":
            transform = transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Resize((self.size, self.size), antialias=None),
                ]
            )
            self.data_loader_str = self.data_loader
            dataset = Food101(".", split="train", transform=transform, download=True)
            self.data_loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
            )

    def _loss_fn(self, x: torch.Tensor) -> float:
        random_t = torch.rand(x.shape[0], device=x.device) * (1.0 - self.eps) + self.eps
        z = torch.randn_like(x)
        mean, std = self.sde.marginal_prob(x, random_t)
        perturbed_data = mean + z * std[:, None, None, None]
        score = self.score_model(perturbed_data, random_t)
        loss = torch.mean(
            torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1, 2, 3))
        )
        return loss

    def train_model(
        self, n_epochs: int = 50, batch_size: int = 32, lr: float = 1e-4
    ) -> None:
        optimizer = Adam(self.score_model.parameters(), lr=lr)
        # tqdm_epoch = tqdm.trange(n_epochs)
        self.losses = []

        for _ in ntqdm(range(n_epochs)):
            avg_loss = 0
            num_items = 0

            for x, _ in self.data_loader:
                x = x.to(self.device)
                loss = self._loss_fn(x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]
            # tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
            self.losses.append(avg_loss / num_items)

            torch.save(self.score_model.state_dict(), "ckpt.pth")

        self.x_size = x.shape

    def load_model(self, path: str = None) -> None:
        if not path:
            path = saved_model_paths[self.data_loader_str][find_sde(self.sde)]

        ckpt = torch.load(path, map_location=self.device)
        self.score_model.load_state_dict(ckpt)

    def _plot_loss(self):
        abs = np.arange(len(self.losses))
        plt.plot(abs, self.losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

    def sampling(self, shape: int = 1) -> torch.Tensor:
        self.sampler.device = self.device
        self.sampler.sde = self.sde
        self.sampler.score_model = self.score_model
        self.sampler.shape = [shape, self.x_size[1], self.x_size[2], self.x_size[3]]

        self.samples = self.sampler.sampling()

        return self.samples

    def direct_sampling(
        self, shape: int = 1, save_evolution: bool = False
    ) -> torch.Tensor:
        self.sampler.device = self.device
        self.sampler.sde = self.sde
        self.sampler.score_model = self.score_model
        self.sampler.shape = [
            shape,
            self.sampler.shape[1],
            self.sampler.shape[2],
            self.sampler.shape[3],
        ]

        if save_evolution:
            self.samples, self.samples_evol = self.sampler.sampling(
                save_evolution=save_evolution
            )
        else:
            self.samples = self.sampler.sampling()

    def plot_samples(
        self, grid: bool = True, comparison: bool = False, title: str = "Samples"
    ) -> None:
        if grid:
            sample_grid = make_grid(
                self.samples, nrow=int(np.sqrt(self.samples.shape[0]))
            )

            plt.figure(figsize=(6, 6))
            plt.axis("off")
            plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0.0, vmax=1.0)
            plt.show()

        elif comparison:
            sample_grid = make_grid(
                self.samples, nrow=int(np.sqrt(self.samples.shape[0]))
            )
            plt.axis("off")
            plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0.0, vmax=1.0)
            plt.title(title)

        else:
            for i in range(self.samples.shape[0]):
                plt.imshow(self.samples[i][0].to("cpu"))
                plt.show()

    def _plot_samples_evol(self, nb_ite: int = 5) -> None:
        l_int = (
            np.array([self.sde.N] * nb_ite)
            - np.logspace(1, np.log10(self.sde.N), nb_ite, base=10.0)
        ).astype(int)
        selected_samples = [self.samples_evol[i][0] for i in l_int]
        sample_grid = make_grid(selected_samples, nrow=len(selected_samples))

        plt.figure(figsize=(6, 6))
        plt.axis("off")
        plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0.0, vmax=1.0)
        plt.show()

    def _get_size_ex(self) -> list:
        for x, _ in self.data_loader:
            self.x_example = x
            break
