import torch
import torch.nn as nn
import math


class Channel(nn.Module):
    def __init__(self, channel_type='AWGN', snr=20, K_factor=1):
        if channel_type not in ['AWGN', 'Rayleigh', 'Rician']:
            raise Exception('Unknown type of channel')
        super(Channel, self).__init__()
        self.channel_type = channel_type
        self.snr = snr
        self.K_factor = K_factor

    def forward(self, z_hat):
        if z_hat.dim() not in {3, 4}:
            raise ValueError('Input tensor must be 3D or 4D')
        
        # if z_hat.dim() == 4:
        #     # k = np.prod(z_hat.size()[1:])
        #     k = torch.prod(torch.tensor(z_hat.size()[1:]))
        #     sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True) / k
        # elif z_hat.dim() == 3:
        #     # k = np.prod(z_hat.size())
        #     k = torch.prod(torch.tensor(z_hat.size()))
        #     sig_pwr = torch.sum(torch.abs(z_hat).square()) / k
            
        if z_hat.dim() == 3:
            z_hat = z_hat.unsqueeze(0)
        
        k = z_hat[0].numel()
        sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True) / k    
        noi_pwr = sig_pwr / (10 ** (self.snr / 10))
        # Generate complex AWGN, splitting power between real and imag parts.
        noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr / 2)

        # --- Fading Application ---
        if self.channel_type == 'Rayleigh':
            # Generate unit-power complex fading coefficient
            hc = torch.randn(2, device=z_hat.device) / math.sqrt(2.0)
            h_real, h_imag = hc[0], hc[1]

            # Decompose signal and apply fading via complex multiplication
            mid_channel = z_hat.size(1) // 2
            z_real, z_imag = z_hat[:, :mid_channel], z_hat[:, mid_channel:]
            y_real = z_real * h_real - z_imag * h_imag
            y_imag = z_real * h_imag + z_imag * h_real
            z_faded = torch.cat((y_real, y_imag), dim=1)

        elif self.channel_type == 'Rician':
            # Rician fading: h = sqrt(K/(K+1))*h_los + sqrt(1/(K+1))*h_nlos
            K_linear = 10**(self.K_factor / 10)

            # Line-of-sight (LOS) component (deterministic, power 1)
            h_los_real = math.sqrt(K_linear / (K_linear + 1))
            h_los_imag = 0.0

            # Non-line-of-sight (NLOS) component (Rayleigh part)
            h_nlos_gain = math.sqrt(1.0 / (K_linear + 1))
            h_nlos = torch.randn(2, device=z_hat.device) / math.sqrt(2.0) * h_nlos_gain
            h_nlos_real, h_nlos_imag = h_nlos[0], h_nlos[1]

            # Total channel coefficient
            h_real = h_los_real + h_nlos_real
            h_imag = h_los_imag + h_nlos_imag

            # Decompose signal and apply fading
            mid_channel = z_hat.size(1) // 2
            z_real, z_imag = z_hat[:, :mid_channel], z_hat[:, mid_channel:]
            y_real = z_real * h_real - z_imag * h_imag
            y_imag = z_real * h_imag + z_imag * h_real
            z_faded = torch.cat((y_real, y_imag), dim=1)

        else:  # AWGN
            z_faded = z_hat

        return z_faded + noise

    def get_channel(self):
        return self.channel_type, self.snr


if __name__ == '__main__':
    # test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")

    awgn_channel = Channel(channel_type='AWGN', snr=10.0).to(device)
    z_hat_awgn = torch.randn(64, 20, 8, 8).to(device)
    output_awgn = awgn_channel(z_hat_awgn)
    print(f"AWGN In: {z_hat_awgn.shape}, Out: {output_awgn.shape}")

    rayleigh_channel = Channel(channel_type='Rayleigh', snr=10).to(device)
    z_hat_rayleigh = torch.randn(10, 20, 8, 8).to(device)
    output_rayleigh = rayleigh_channel(z_hat_rayleigh)
    print(f"Rayleigh In: {z_hat_rayleigh.shape}, Out: {output_rayleigh.shape}")

    rician_channel = Channel(channel_type='Rician', snr=10, K_factor=1).to(device)
    output_rician = rician_channel(z_hat_rayleigh)
    print(f"Rician In: {z_hat_rayleigh.shape}, Out: {output_rician.shape}")
