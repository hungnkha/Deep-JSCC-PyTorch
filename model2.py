# -*- coding: utf-8 -*-
"""
Created on Tue Dec  11:00:00 2023

@author: chun
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from channel import Channel
from binary_converter import bit2float, float2bit


def modulate(bits, M):
    """
    General M-QAM modulation (supports BPSK, QPSK, 16-QAM, 64-QAM, ...)

    Args:
        bits: torch.Tensor of shape [N, log2(M)]
        M: modulation order (must be power of 2 and square, e.g., 2, 4, 16, 64)
    
    Returns:
        I, Q: real and imaginary parts of the modulated symbols
    """
    assert (M & (M - 1) == 0) and M >= 2, "M must be power of 2 and >= 2"
    k = int(math.log2(M))  # bits per symbol

    if M == 2:  # BPSK
        symbols = 2 * bits - 1  # 0 → -1, 1 → +1
        return symbols.view(-1), torch.zeros_like(symbols.view(-1))

    assert k % 2 == 0, "Only square M-QAM supported (even number of bits per symbol)"

    bits = bits.view(-1, k)
    m_side = int(2 ** (k // 2))  # sqrt(M), e.g., 4 for 16-QAM

    # Convert bits to symbol index per axis
    b_I = bits[:, :k//2].int()
    b_Q = bits[:, k//2:].int()

    power_vec = (2 ** torch.arange(k//2 - 1, -1, -1, device=bits.device)).to(torch.int32)
    #I_idx = b_I.matmul(power_vec)
    #Q_idx = b_Q.matmul(power_vec)

    I_idx = (b_I*power_vec).sum(dim=1)
    Q_idx = (b_Q*power_vec).sum(dim=1)

    # Map to constellation points (-m_side+1, ..., m_side-1)
    I_sym = 2 * I_idx - (m_side - 1)
    Q_sym = 2 * Q_idx - (m_side - 1)

    # Normalize average symbol energy to 1
    norm_factor = math.sqrt((2 / 3) * (M - 1))
    I_sym = I_sym.float() / norm_factor
    Q_sym = Q_sym.float() / norm_factor

    return I_sym, Q_sym


def demodulate(i, q, M):
    """
    General M-QAM hard-decision demodulation

    Args:
        i, q: real and imaginary parts of received symbols (flattened tensors)
        M: modulation order (2, 4, 16, 64, ...)
    
    Returns:
        bits: [N, log2(M)] tensor of hard-decoded bits
    """
    assert (M & (M - 1) == 0) and M >= 2, "M must be power of 2 and >= 2"
    k = int(math.log2(M))

    if M == 2:  # BPSK
        bits = (i > 0).int().view(-1, 1)
        return bits

    assert k % 2 == 0, "Only square M-QAM supported (even k)"

    m_side = int(2 ** (k // 2))
    norm_factor = math.sqrt((2 / 3) * (M - 1))

    # De-normalize
    i = i * norm_factor
    q = q * norm_factor

    # Clamp to valid range
    i_clamped = torch.clamp(i, -m_side + 1, m_side - 1)
    q_clamped = torch.clamp(q, -m_side + 1, m_side - 1)

    # Round to nearest symbol index
    i_idx = ((i_clamped + (m_side - 1)) / 2).round().int()
    q_idx = ((q_clamped + (m_side - 1)) / 2).round().int()

    # Ensure valid symbol indices
    i_idx = torch.clamp(i_idx, 0, m_side - 1)
    q_idx = torch.clamp(q_idx, 0, m_side - 1)

    def int_to_bits(x, num_bits):
        return ((x.unsqueeze(-1) >> torch.arange(num_bits - 1, -1, -1, device=x.device)) & 1).int()

    bits_i = int_to_bits(i_idx, k // 2)
    bits_q = int_to_bits(q_idx, k // 2)

    bits = torch.cat([bits_i, bits_q], dim=-1)
    return bits


def add_awgn(i, q, snr_db):
    snr_linear = 10 ** (snr_db / 10)
    power = (i ** 2 + q ** 2).mean()
    noise_power = power / snr_linear
    noise_std = torch.sqrt(noise_power / 2)

    i_noisy = i + noise_std * torch.randn_like(i)
    q_noisy = q + noise_std * torch.randn_like(q)
    return i_noisy, q_noisy


def add_rayleigh(i, q, snr_db):
    # Create complex input
    x = i + 1j * q

    # Rayleigh fading channel: real and imag parts ~ N(0, 0.5)
    h_real = torch.randn_like(i) / torch.sqrt(torch.tensor(2.0))
    h_imag = torch.randn_like(q) / torch.sqrt(torch.tensor(2.0))
    h = h_real + 1j * h_imag  # complex fading

    # Apply Rayleigh fading
    y = h * x

    # Compute signal power
    power = (y.real**2 + y.imag**2).mean()

    # Compute noise power from SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = power / snr_linear
    noise_std = torch.sqrt(noise_power / 2)

    # Add AWGN
    noise_real = noise_std * torch.randn_like(i)
    noise_imag = noise_std * torch.randn_like(q)

    y_noisy = y + noise_real + 1j * noise_imag

    # Return noisy i and q components
    return y_noisy.real, y_noisy.imag



def toeplitz(c, r=None):
    if r is None:
        r = c
    c = c.to(device=r.device)
    r = r.to(device=c.device)
    assert c[0].item() == r[0].item(), "First element of c and r must be the same"
    n = c.numel()
    m = r.numel()
    idx = torch.arange(n, device=c.device).unsqueeze(1) - torch.arange(m, device=c.device).unsqueeze(0)
    return torch.where(
        idx >= 0,
        c[idx.clamp(min=0)],
        r[(-idx).clamp(min=0)]
    )

def add_rayleigh_with_ar_nosave(velocity, carrier_fre, bandwidth, i, q, snr_db):
# velocity (m/s), carrier_fre (Hz), bandwidth (khz)

    P = 6
    v = velocity #m/s
    fc = carrier_fre #hz
    fm = v*fc/3/1e8 # Doppler frequency
    #fm = 150
    # BW = 2000 # kHz
    M_QAM = 4 # 4-QAM
    fs = bandwidth*math.log2(M_QAM) #ksps

    epselonn = 1e-6 # Small bias for numerical stability


    device = i.device
    M = i.numel()
    
    # Complex input signal
    x = i + 1j * q
    x = x.to(torch.complex64)
    
    # Step 1: Autocorrelation vector using Bessel J0
    p_idx = torch.arange(0, P + 1, dtype=torch.float32, device=device)
    arg = 2 * math.pi * fm * p_idx / (fs * 1000)
    vector_corr = torch.special.bessel_j0(arg)
    
    # Step 2: Toeplitz autocorrelation matrix + bias
    toeplitz_col = vector_corr[:P]
    toeplitz_row = vector_corr[:P]
    R = toeplitz(toeplitz_col, toeplitz_row) + epselonn * torch.eye(P, device=device)
    
    # Step 3: Solve Yule-Walker
    r = vector_corr[1:P + 1]
    AR_parameters = -torch.linalg.solve(R, r)
    
    # Step 4: Noise variance
    sigma_u = R[0, 0] + torch.dot(r, AR_parameters)
    if sigma_u <= 0:
        raise ValueError(f"Noise variance sigma_u={sigma_u} <= 0, check parameters.")
    
    # Step 5: Generate complex white Gaussian noise
    KKK = 2000
    total_len = M + KKK
    noise_std = torch.sqrt(sigma_u / 2)
    
    noise_real = torch.randn(total_len, device=device) * noise_std
    noise_imag = torch.randn(total_len, device=device) * noise_std
    noise = noise_real + 1j * noise_imag
    noise = noise.to(torch.complex64)
    
    # Step 6: AR filter for channel coefficients h
    h = torch.zeros(total_len, dtype=torch.complex64, device=device)
    AR_params_c = AR_parameters.to(torch.complex64)
    for n in range(P, total_len):
        past = h[n-P:n].flip(0)
        h[n] = noise[n] - torch.dot(AR_params_c, past)
    h = h[KKK:]  # Remove warm-up

    h_power = (h.real ** 2 + h.imag ** 2).mean()

    h = h / torch.sqrt(h_power)
 

    # Step 7: Apply fading to input
    y = x * h
    
    # Step 8: Compute power and noise for AWGN
    power = (y.real**2 + y.imag**2).mean()
    snr_linear = 10 ** (snr_db / 10)
    noise_power = power / snr_linear
    noise_std_awgn = torch.sqrt(noise_power / 2)
    
    noise_awgn_real = noise_std_awgn * torch.randn(M, device=device)
    noise_awgn_imag = noise_std_awgn * torch.randn(M, device=device)
    
    y_noisy = y + noise_awgn_real + 1j * noise_awgn_imag
    
    return y_noisy.real, y_noisy.imag


import os


def add_rayleigh_with_ar(velocity, carrier_fre, bandwidth, i, q, snr_db, M_QAM):
    # Constants
    P = 6
    v = velocity
    fc = carrier_fre
    fm = v * fc / 3 / 1e8
    # M_QAM = 4
    fs = bandwidth * math.log2(M_QAM)  # ksps
    epselonn = 1e-6

    # Input info
    device = i.device
    M = i.numel()

    # Complex input signal
    x = i + 1j * q
    x = x.to(torch.complex64)

    # Cache key and file path
    folder_name = f"/home/MATLAB_DATA/TiNguyen/SentryJSCC/File_{int(v)}_{int(fc)}_{int(bandwidth)}_{M}_{M_QAM}"
    os.makedirs(folder_name, exist_ok=True)
    param_path = os.path.join(folder_name, "ar_cache.pt")

    # Try loading parameters
    if os.path.exists(param_path):
        cache = torch.load(param_path, map_location=device)
        AR_parameters = cache["AR_parameters"]
        R = cache["R"]
        r = cache["r"]
        sigma_u = cache["sigma_u"]
    else:
      
        # Step 1: Autocorrelation vector using Bessel J0
        p_idx = torch.arange(0, P + 1, dtype=torch.float32, device=device)
        arg = 2 * math.pi * fm * p_idx / (fs * 1000)
        vector_corr = torch.special.bessel_j0(arg)
        
        # Step 2: Toeplitz autocorrelation matrix + bias
        toeplitz_col = vector_corr[:P]
        toeplitz_row = vector_corr[:P]
        R = toeplitz(toeplitz_col, toeplitz_row) + epselonn * torch.eye(P, device=device)
        
        # Step 3: Solve Yule-Walker
        r = vector_corr[1:P + 1]
        AR_parameters = -torch.linalg.solve(R, r)
        
        # Step 4: Noise variance
        sigma_u = R[0, 0] + torch.dot(r, AR_parameters)
        if sigma_u <= 0:
            raise ValueError(f"Noise variance sigma_u={sigma_u} <= 0, check parameters.")

        # Save all
        torch.save({
            "AR_parameters": AR_parameters,
            "R": R,
            "r": r,
            "sigma_u": sigma_u
        }, param_path)

    # Step 5: Generate white Gaussian noise
    KKK = 2000
    total_len = M + KKK
    noise_std = torch.sqrt(sigma_u / 2)

    noise_real = torch.randn(total_len, device=device) * noise_std
    noise_imag = torch.randn(total_len, device=device) * noise_std
    noise = (noise_real + 1j * noise_imag).to(torch.complex64)

    # Step 6: AR filter to generate fading channel h
    h = torch.zeros(total_len, dtype=torch.complex64, device=device)
    AR_params_c = AR_parameters.to(torch.complex64)


    # for n in range(P, total_len):
    #     past = h[n - P:n].flip(0)
    #     h[n] = noise[n] - torch.dot(AR_params_c, past)
    # h = h[KKK:]

    # Ensure inputs are shaped correctly for conv1d
    noise_reshaped = noise.view(1, 1, -1)  # shape (batch=1, channel=1, time)
    kernel = AR_params_c.flip(0).view(1, 1, -1)  # shape (out_channels=1, in_channels=1, kernel_size=P)
    # Pad the front with P zeros to emulate initial condition
    padded = F.pad(noise_reshaped, (P-1, 0))
    # Apply 1D convolution to simulate the AR process
    filtered = F.conv1d(padded, kernel)
    # print(f"noise_reshaped size: {noise_reshaped.shape}")  
    # print(f"AR kernel size: {kernel.shape}")            # (1, 1, P)
    # print(f"Padded noise size: {padded.shape}")         # (1, 1, total_len + P - 1)
    # print(f"Filtered output size: {filtered.shape}")    # (1, 1, total_len)
    # Subtract the AR component from noise
    h = noise_reshaped- filtered  # shape (1, 1, total_len)
    h = h.view(-1)[KKK:]  # flatten and remove warm-up



    h_power = (h.real ** 2 + h.imag ** 2).mean()
    h = h / torch.sqrt(h_power)

    # Step 7: Apply fading
    y = x * h

    # Step 8: Add AWGN
    power = (y.real**2 + y.imag**2).mean()
    snr_linear = 10 ** (snr_db / 10)
    noise_power = power / snr_linear
    noise_std_awgn = torch.sqrt(noise_power / 2)

    noise_awgn_real = noise_std_awgn * torch.randn(M, device=device)
    noise_awgn_imag = noise_std_awgn * torch.randn(M, device=device)
    y_noisy = y + noise_awgn_real + 1j * noise_awgn_imag

    return y_noisy.real, y_noisy.imag



def ratio2filtersize(x: torch.Tensor, ratio):
    if x.dim() == 4:
        # before_size = np.prod(x.size()[1:])
        before_size = torch.prod(torch.tensor(x.size()[1:]))
    elif x.dim() == 3:
        # before_size = np.prod(x.size())
        before_size = torch.prod(torch.tensor(x.size()))
    else:
        raise Exception('Unknown size of input')
    encoder_temp = _Encoder(is_temp=True)
    z_temp = encoder_temp(x)
    # c = before_size * ratio / np.prod(z_temp.size()[-2:])
    c = before_size * ratio / torch.prod(torch.tensor(z_temp.size()[-2:]))
    return int(c)


class _ConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(_ConvWithPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU()

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x


class _TransConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activate=nn.PReLU(), padding=0, output_padding=0):
        super(_TransConvWithPReLU, self).__init__()
        self.transconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.activate = activate
        if activate == nn.PReLU():
            nn.init.kaiming_normal_(self.transconv.weight, mode='fan_out',
                                    nonlinearity='leaky_relu')
        else:
            nn.init.xavier_normal_(self.transconv.weight)

    def forward(self, x):
        x = self.transconv(x)
        x = self.activate(x)
        return x


class _Encoder(nn.Module):
    def __init__(self, c=1, is_temp=False, P=1):
        super(_Encoder, self).__init__()
        self.is_temp = is_temp
        # self.imgae_normalization = _image_normalization(norm_type='nomalization')
        self.conv1 = _ConvWithPReLU(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv2 = _ConvWithPReLU(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv3 = _ConvWithPReLU(in_channels=32, out_channels=32,
                                    kernel_size=5, padding=2)  # padding size could be changed here
        self.conv4 = _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv5 = _ConvWithPReLU(in_channels=32, out_channels=2*c, kernel_size=5, padding=2)
        self.norm = self._normlizationLayer(P=P)

    @staticmethod
    def _normlizationLayer(P=1):
        def _inner(z_hat: torch.Tensor):
            if z_hat.dim() == 4:
                batch_size = z_hat.size()[0]
                # k = np.prod(z_hat.size()[1:])
                k = torch.prod(torch.tensor(z_hat.size()[1:]))
            elif z_hat.dim() == 3:
                batch_size = 1
                # k = np.prod(z_hat.size())
                k = torch.prod(torch.tensor(z_hat.size()))
            else:
                raise Exception('Unknown size of input')
            # k = torch.tensor(k)
            z_temp = z_hat.reshape(batch_size, 1, 1, -1)
            z_trans = z_hat.reshape(batch_size, 1, -1, 1)
            tensor = torch.sqrt(P * k) * z_hat / torch.sqrt((z_temp @ z_trans))
            if batch_size == 1:
                return tensor.squeeze(0)
            return tensor
        return _inner

    def forward(self, x):
        # x = self.imgae_normalization(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if not self.is_temp:
            x = self.conv5(x)
            x = self.norm(x)
        return x


class _Decoder(nn.Module):
    def __init__(self, c=1):
        super(_Decoder, self).__init__()
        # self.imgae_normalization = _image_normalization(norm_type='denormalization')
        self.tconv1 = _TransConvWithPReLU(
            in_channels=2*c, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv2 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv3 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv4 = _TransConvWithPReLU(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.tconv5 = _TransConvWithPReLU(
            in_channels=16, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1,activate=nn.Sigmoid())
        # may be some problems in tconv4 and tconv5, the kernal_size is not the same as the paper which is 5

    def forward(self, x):
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        # x = self.imgae_normalization(x)
        return x


class ModChan:
    def __init__(self, num_e_bits, num_m_bits, M, SNRdB, channel_type):
        assert (M & (M - 1) == 0), "M must be a power of 2"  # check if M is power of 2
        self.num_e_bits = num_e_bits
        self.num_m_bits = num_m_bits
        self.bias = 2**(self.num_e_bits-1)-1
        self.n_bits = 1 + self.num_e_bits + self.num_m_bits 
        self.M = M
        self.k = int(math.log2(M))  # bits per symbol
        self.SNRdB = SNRdB
        self.channel_type = channel_type
        


    def __call__(self, z):
        z_flat = z.view(-1)

        # Quantize latent vector to bitstream

        bits = float2bit(z_flat, self.num_e_bits, self.num_m_bits, self.bias)

        bitstream = bits.view(-1)  # Flatten to 1D

        # Pad bitstream to make it divisible by k
        remainder = bitstream.numel() % self.k
        if remainder != 0:
            pad_len = self.k - remainder
            bitstream = F.pad(bitstream, (0, pad_len))

        # Group bits for modulation
        bit_groups = bitstream.view(-1, self.k)

        # Modulate into I/Q symbols
        I, Q = modulate(bit_groups, self.M)

        if self.channel_type=='AWGN':
            # Add AWGN noise
            I_noisy, Q_noisy = add_awgn(I, Q, self.SNRdB)
        else: # rayleigh
            #I_noisy, Q_noisy = add_rayleigh(I, Q, self.SNRdB)
            velocity = 15 # m/s
            carrier_fre = 18*1e9 #Hz
            bandwidth = 2000 # kHz
            I_noisy, Q_noisy = add_rayleigh_with_ar(velocity, carrier_fre, bandwidth, I, Q, self.SNRdB, self.M)


        # Demodulate to recover bits
        bits_demod = demodulate(I_noisy, Q_noisy, self.M).float()
        bitstream_recovered = bits_demod.view(-1)[:bits.numel()]  # Trim padding

        # Reshape to original bit shape
        bits_recovered = bitstream_recovered.view(-1, self.n_bits)

        # Convert bits back to float values

        z_recovered_flat = bit2float(bits_recovered, self.num_e_bits, self.num_m_bits, self.bias)

        # Reshape to original shape
        z_recovered = z_recovered_flat.view(z.size())
        return z_recovered



class DeepJSCC(nn.Module):
    def __init__(self, c, channel_type='AWGN', snr=1 , M = 4, num_e_bits = 8, num_m_bits = 23 ):
        super(DeepJSCC, self).__init__()
        self.num_e_bits = num_e_bits
        self.num_m_bits = num_m_bits
        self.bias = 2**(self.num_e_bits-1)-1
        self.n_bits = 1 + self.num_e_bits + self.num_m_bits 

        self.encoder = _Encoder(c=c)
        if snr is not None:
            self.channel = Channel(channel_type, snr)
        self.decoder = _Decoder(c=c)
        self.ModChan = ModChan(num_e_bits, num_m_bits, M, snr, channel_type)
        
    def Enc(self, x):
        z = self.encoder(x)
        return z
    
    def Dec(self, z):
        x_hat = self.decoder(z)
        return x_hat
    

    def Chan(self, z):
        z = self.ModChan(z)
        return z
    
    # def Chan(self, z):
    #     if hasattr(self, 'channel') and self.channel is not None:
    #         z = self.channel(z)
    #     return z
    
    # def forward(self, x):
    #     z = self.encoder(x)
    #     if hasattr(self, 'channel') and self.channel is not None:
    #         z = self.channel(z)
    #     x_hat = self.decoder(z)
    #     return x_hat
    
    def forward(self, x):
        z = self.encoder(x)
        z = self.ModChan(z)
        x_hat = self.decoder(z)
        return x_hat

    def change_channel(self, channel_type='AWGN', snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = Channel(channel_type, snr)

    def get_channel(self):
        if hasattr(self, 'channel') and self.channel is not None:
            return self.channel.get_channel()
        return None

    def loss(self, prd, gt):
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(prd, gt)

        #epsilon = 1e-6
        #error = gt - prd
        #loss = torch.sum(torch.log(error ** 2 + epsilon))
        return loss
    



if __name__ == '__main__':
    model = DeepJSCC(c=20, channel_type='AWGN', snr=20 , M = 2, num_e_bits = 8, num_m_bits = 23)
    print(model)
    x = torch.rand(1, 3, 128, 128)
    y = model(x)
    print(y.size())
    print(y)
    print(model.encoder.norm)
    print(model.encoder.norm(y))
    print(model.encoder.norm(y).size())
    print(model.encoder.norm(y).size()[1:])
