import numpy as np

def optimized_noise_schedule(timesteps):
    beta_start, beta_end = 1e-4, 0.02
    return np.linspace(beta_start, beta_end, timesteps)

class DiffusionScheduler:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = optimized_noise_schedule(timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = np.cumprod(self.alphas, axis=0)
