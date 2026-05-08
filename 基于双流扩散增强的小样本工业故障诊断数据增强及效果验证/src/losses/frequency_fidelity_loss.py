import torch
import torch.nn as nn
import torch.fft


class FrequencyFidelityLoss(nn.Module):
    def __init__(self, fs=12000, target_frequencies=None):
        super().__init__()
        self.fs = fs
        if target_frequencies is None:
            self.target_frequencies = torch.tensor([25.0, 79.4, 120.6, 800.0])
        else:
            self.target_frequencies = torch.tensor(target_frequencies)
    
    def forward(self, generated, real):
        generated_psd = self.calculate_psd(generated)
        real_psd = self.calculate_psd(real)
        
        psd_mse = nn.MSELoss()(generated_psd, real_psd)
        
        return psd_mse
    
    def calculate_psd(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        n = x.shape[-1]
        fft_result = torch.fft.fft(x)
        psd = torch.abs(fft_result) ** 2 / n
        
        freqs = torch.fft.fftfreq(n, 1/self.fs)
        positive_mask = freqs >= 0
        psd = psd[:, positive_mask]
        freqs = freqs[positive_mask]
        
        target_psd = []
        for freq in self.target_frequencies:
            idx = torch.argmin(torch.abs(freqs - freq))
            target_psd.append(psd[:, idx])
        
        return torch.stack(target_psd, dim=1)