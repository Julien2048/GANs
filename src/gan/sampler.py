import torch
from tqdm.notebook import tqdm as ntqdm
import numpy as np
from scipy import integrate


# @title Euler-Maruyama Direct Sampling
class EulerMaruyamaSampler:
    def __init__(
        self,
        sde=None,
        score_model=None,
        corrector: bool = False,
        shape: list = [1, 1, 28, 28],
        eps: float = 1e-3,
        device: float = "cuda",
    ) -> None:
        self.sde = sde
        # self.rsde = sde.reverse(score_model)
        self.score_model = score_model
        self.corrector = corrector
        self.shape = shape
        self.eps = eps
        self.device = device

    def sampling(self, save_evolution: bool = False) -> torch.Tensor:
        if save_evolution:
            sampler_evol = []

        init_x = self.sde.prior_sampling(self.shape).to(self.device)
        time_steps = torch.linspace(1.0, self.eps, self.sde.N, device=self.device)
        step_size = time_steps[0] - time_steps[1]
        x = init_x
        with torch.no_grad():
            for time_step in ntqdm(time_steps):
                x = x.to(self.device)
                batch_time_step = torch.ones(x.shape[0], device=self.device) * time_step
                _, g = self.sde.sde(x, batch_time_step)
                mean_x = (
                    x
                    + (g**2)[:, None, None, None]
                    * self.score_model(x, batch_time_step)
                    * step_size
                )
                x = mean_x + torch.sqrt(step_size) * g[
                    :, None, None, None
                ] * torch.randn_like(x)
                if save_evolution:
                    sampler_evol.append(mean_x)

        self.sampler = mean_x

        if save_evolution:
            return self.sampler, sampler_evol
        else:
            return self.sampler


class EulerMaruyamaSamplerCorrector:
    def __init__(
        self,
        sde=None,
        score_model=None,
        corrector: bool = False,
        shape: list = [1, 1, 28, 28],
        eps: float = 1e-3,
        num_steps_cor: int = 10,
        device: str = "cuda",
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
        time_steps = torch.linspace(1.0, self.eps, self.sde.N, device=self.device)
        step_size = time_steps[0] - time_steps[1]
        x = init_x
        with torch.no_grad():
            for time_step in ntqdm(time_steps):
                x = x.to(self.device)
                batch_time_step = torch.ones(x.shape[0], device=self.device) * time_step

                for _ in range(self.num_steps_cor):
                    grad = self.score_model(x, batch_time_step)
                    grad_norm = torch.norm(
                        grad.reshape(grad.shape[0], -1), dim=-1
                    ).mean()
                    noise_norm = np.sqrt(np.prod(x.shape[1:]))
                    langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
                    x = (
                        x
                        + langevin_step_size * grad
                        + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
                    )

                _, g = self.sde.sde(x, batch_time_step)
                mean_x = (
                    x
                    + (g**2)[:, None, None, None]
                    * self.score_model(x, batch_time_step)
                    * step_size
                )
                x = mean_x + torch.sqrt(step_size) * g[
                    :, None, None, None
                ] * torch.randn_like(x)

        self.sampler = mean_x
        return self.sampler


# @title ODE Sampling
class ODESampler:
    def __init__(
        self,
        sde=None,
        score_model=None,
        shape: list = [1, 1, 28, 28],
        eps: float = 1e-3,
        device: str = "cpu",
        error_tolerance=1e-5,
    ) -> None:
        self.sde = sde
        self.score_model = score_model
        self.eps = eps
        self.device = device
        self.shape = shape
        self.error_tolerance = error_tolerance

    def _score_eval_wrapper(self, sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = torch.tensor(sample, device=self.device, dtype=torch.float32).reshape(
            self.shape
        )
        time_steps = torch.tensor(
            time_steps, device=self.device, dtype=torch.float32
        ).reshape((sample.shape[0],))
        with torch.no_grad():
            score = self.score_model(sample, time_steps)

        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def _ode_func(self, t, x):
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((self.shape[0],)) * t
        _, g = self.sde.sde(torch.tensor(x), torch.tensor(time_steps))
        return -0.5 * (g[0] ** 2) * self._score_eval_wrapper(x, time_steps)

    def sampling(self) -> torch.Tensor:
        init_x = self.sde.prior_sampling(self.shape).to(self.device)

        res = integrate.solve_ivp(
            self._ode_func,
            (1.0, self.eps),
            init_x.reshape(-1).cpu().numpy(),
            rtol=self.error_tolerance,
            atol=self.error_tolerance,
            method="RK45",
        )
        print(f"Number of function evaluations: {res.nfev}")
        x = torch.tensor(res.y[:, -1], device=self.device).reshape(self.shape)

        return x
