import torch
from tqdm.notebook import tqdm as ntqdm
import numpy as np

#@title Euler-Maruyama Direct Sampling

class EulerMaruyamaSampler():
    def __init__(
        self,
        sde = None,
        score_model = None, 
        corrector: bool = False,
        shape: list = [1, 1, 28, 28],
        eps: float = 1e-3,
        device: float = 'cuda'
    ) -> None:
        self.sde = sde
        # self.rsde = sde.reverse(score_model)
        self.score_model = score_model
        self.corrector = corrector
        self.shape = shape
        self.eps = eps
        self.device = device

    def sampling(self, save_evolution: bool = False) -> torch.Tensor:

        # t = torch.ones(batch_size, device=device)
        # init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
        #   * marginal_prob_std(t)[:, None, None, None]

        if save_evolution:
            sampler_evol = []

        init_x = self.sde.prior_sampling(self.shape).to(self.device)
        time_steps = torch.linspace(1., self.eps, self.sde.N, device=self.device)
        step_size = time_steps[0] - time_steps[1]
        x = init_x
        with torch.no_grad():
          for time_step in ntqdm(time_steps):
            x = x.to(self.device)
            batch_time_step = torch.ones(x.shape[0], device=self.device) * time_step
            f, g = self.sde.sde(x, batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * self.score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
            if save_evolution:
                sampler_evol.append(mean_x)      
        
        self.sampler = mean_x
        return self.sampler if not(save_evolution) else self.sampler, sampler_evol


class EulerMaruyamaSamplerCorrector():
    def __init__(
        self,
        sde = None,
        score_model = None, 
        corrector: bool = False,
        shape: list = [1, 1, 28, 28],
        eps: float = 1e-3,
        num_steps_cor: int = 10,
        device: str = 'cuda'
    ) -> None:
        self.sde = sde
        # self.rsde = sde.reverse(score_model)
        self.score_model = score_model
        self.corrector = corrector
        self.shape = shape
        self.eps = eps
        self.num_steps_cor = num_steps_cor
        self.device = device

    def sampling(self, snr: float = 0.16) -> torch.Tensor:
        init_x = self.sde.prior_sampling(self.shape).to(self.device)
        time_steps = torch.linspace(1., self.eps, self.sde.N, device=self.device)
        step_size = time_steps[0] - time_steps[1]
        x = init_x
        with torch.no_grad():
            for time_step in ntqdm(time_steps):
                x = x.to(self.device)
                batch_time_step = torch.ones(x.shape[0], device=self.device) * time_step

                for n_cor in range(self.num_steps_cor):

                    grad = self.score_model(x, batch_time_step)
                    grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                    noise_norm = np.sqrt(np.prod(x.shape[1:]))
                    langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
                    x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

                f, g = self.sde.sde(x, batch_time_step)
                mean_x = x + (g**2)[:, None, None, None] * self.score_model(x, batch_time_step) * step_size
                x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      
              
        self.sampler = mean_x
        return self.sampler