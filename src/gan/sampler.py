import torch
from tqdm.notebook import tqdm as ntqdm

class EulerMaruyamaSampler():
    def __init__(
        self,
        sde = None,
        score_model = None, 
        corrector: bool = False,
        shape: list = [1, 1, 28, 28],
        eps: float = 1e-3,
        device: str = 'cuda'
    ) -> None:
        self.sde = sde
        # self.rsde = sde.reverse(score_model)
        self.score_model = score_model
        self.corrector = corrector
        self.shape = shape
        self.eps = eps
        self.device = device

    def sampling(self) -> torch.Tensor:

        # t = torch.ones(batch_size, device=device)
        # init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
        #   * marginal_prob_std(t)[:, None, None, None]

        init_x = self.sde.prior_sampling(self.shape).to(self.device)

        time_steps = torch.linspace(1., self.eps, self.sde.N, device=self.device)
        step_size = time_steps[0] - time_steps[1]
        x = init_x
        with torch.no_grad():
          for time_step in ntqdm(time_steps):
            x = x.to(self.device)
            batch_time_step = torch.ones(x.shape[0], device=self.device) * time_step
            f, g = self.sde.sde(x, batch_time_step)
            # g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * self.score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      
        # Do not include any noise in the last sampling step.
        
        self.sampler = mean_x
        return self.sampler