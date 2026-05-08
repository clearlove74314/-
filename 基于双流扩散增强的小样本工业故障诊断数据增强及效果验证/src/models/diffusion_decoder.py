import torch
import torch.nn as nn


class DiffusionDecoder(nn.Module):
    def __init__(self, input_length=2048, timesteps=200, noise_schedule='cosine',
                 beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self.input_length = input_length
        self.timesteps = timesteps
        
        if noise_schedule == 'cosine':
            self.betas = self.cosine_beta_schedule(timesteps)
        else:
            self.betas = self.linear_beta_schedule(timesteps, beta_start, beta_end)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.model = self.build_unet()
    
    def linear_beta_schedule(self, timesteps, beta_start, beta_end):
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def build_unet(self):
        return nn.Sequential(
            nn.Conv1d(2, 64, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 1, 7, padding=3)
        )
    
    def add_noise(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].to(x_start.device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(x_start.device)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def forward(self, x, t, condition):
        t_emb = self.get_timestep_embedding(t).unsqueeze(-1).repeat(1, 1, x.shape[-1])
        x = torch.cat([x, t_emb], dim=1)
        
        if condition is not None:
            cond_emb = condition.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = torch.cat([x, cond_emb], dim=1)
        
        return self.model(x)
    
    def get_timestep_embedding(self, t):
        half_dim = 32
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = t.float() * emb.to(t.device)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def sample(self, condition, num_samples=1, device='cuda'):
        x = torch.randn(num_samples, 1, self.input_length).to(device)
        
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.tensor([t] * num_samples).to(device)
            predicted_noise = self(x, t_tensor, condition)
            
            alpha_t = self.alphas[t].to(device)
            alpha_cumprod_t = self.alphas_cumprod[t].to(device)
            alpha_cumprod_prev = self.alphas_cumprod_prev[t].to(device)
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)
            if t > 0:
                x = x + torch.sqrt(1 - alpha_cumprod_prev) * noise
        
        return x.squeeze(1)