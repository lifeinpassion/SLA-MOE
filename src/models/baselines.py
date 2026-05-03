"""
Baseline models for EEG denoising comparison.

This module contains implementations of:
- EEGDnoiseNet: Deep convolutional network for EEG denoising
- EEGDnet: Encoder-decoder network for EEG artifact removal
- RNNEEG: RNN-based EEG denoising
- ResNetEEG: ResNet-based EEG denoising
- Traditional filters: Wiener, LMS, RLS, Kalman
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import wiener
import logging
from typing import Optional, Tuple

from ..utils.seed_utils import set_all_seeds


# =============================================================================
# Deep Learning Baselines
# =============================================================================

class EEGDnoiseNet(nn.Module):
    """
    EEGDnoiseNet: Deep Convolutional Network for EEG Denoising.

    Based on the architecture from:
    "EEGDenoiseNet: A Benchmark Dataset for Deep Learning Solutions of EEG Denoising"

    Architecture:
    - Multiple 1D convolutional layers with residual connections
    - Batch normalization and dropout for regularization
    - Skip connections for gradient flow
    """

    def __init__(self, input_size: int = 512, num_channels: int = 1,
                 hidden_channels: int = 64, num_layers: int = 10):
        super().__init__()
        self.input_size = input_size
        self.num_channels = num_channels

        # Input projection
        self.input_conv = nn.Conv1d(num_channels, hidden_channels, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm1d(hidden_channels)

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.res_blocks.append(
                self._make_res_block(hidden_channels, hidden_channels, dilation=2 ** (i % 4))
            )

        # Output projection
        self.output_conv = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels // 2, num_channels, kernel_size=1)
        )

    def _make_res_block(self, in_channels: int, out_channels: int, dilation: int = 1):
        """Create a residual block with dilated convolutions."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3,
                     padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len) or (batch, 1, seq_len)

        Returns:
            Denoised signal of same shape as input
        """
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension

        # Input projection
        h = F.relu(self.input_bn(self.input_conv(x)))

        # Residual blocks
        for res_block in self.res_blocks:
            residual = h
            h = res_block(h)
            h = F.relu(h + residual)

        # Output projection
        out = self.output_conv(h)

        return out.squeeze(1)


class EEGDnet(nn.Module):
    """
    EEGDnet: Encoder-Decoder Network for EEG Artifact Removal.

    Based on the architecture from:
    "EEGDnet: Fusing non-local and local self-similarity for EEG signal denoising"

    Architecture:
    - Encoder with downsampling convolutions
    - Bottleneck with attention mechanism
    - Decoder with upsampling convolutions
    - Skip connections between encoder and decoder
    """

    def __init__(self, input_size: int = 512, base_channels: int = 32):
        super().__init__()
        self.input_size = input_size

        # Encoder
        self.enc1 = self._conv_block(1, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool1d(2)

        # Bottleneck with self-attention
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_channels * 8, base_channels * 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 16),
            nn.ReLU(),
            nn.Conv1d(base_channels * 16, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 8),
            nn.ReLU()
        )

        # Simple attention (non-local block simplified)
        # Output channels MUST match `b`'s channels (base_channels * 8) so that
        # `b = b * att` is shape-compatible. Previously this was * 4 which
        # caused the runtime shape mismatch error.
        self.attention = nn.Sequential(
            nn.Conv1d(base_channels * 8, base_channels * 8, kernel_size=1),
            nn.Softmax(dim=-1)
        )

        # Decoder
        # Each dec_N block receives torch.cat([d_N, e_N], dim=1), so its input
        # channels MUST equal `up_N.out_channels + enc_N.out_channels`. The
        # original code had these mismatched on dec4/dec3/dec2 (off by a factor
        # that depended on the encoder's deeper width); the runtime error
        # surfaces as "Given groups=1, weight of size [...], expected input to
        # have N channels, but got M instead."
        self.up4 = nn.ConvTranspose1d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(base_channels * 4 + base_channels * 8, base_channels * 4)

        self.up3 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(base_channels * 2 + base_channels * 4, base_channels * 2)

        self.up2 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base_channels + base_channels * 2, base_channels)

        self.up1 = nn.ConvTranspose1d(base_channels, base_channels, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)  # already correct: 1+1

        # Output
        self.output = nn.Conv1d(base_channels, 1, kernel_size=1)

    def _conv_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with U-Net style skip connections."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Apply attention
        att = self.attention(b)
        b = b * att

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self._match_size(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self._match_size(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self._match_size(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self._match_size(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.output(d1)

        # Match output size to input
        if out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, size=x.shape[-1], mode='linear', align_corners=False)

        return out.squeeze(1)

    def _match_size(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Match tensor size using interpolation."""
        if x.shape[-1] != target.shape[-1]:
            x = F.interpolate(x, size=target.shape[-1], mode='linear', align_corners=False)
        return x


class RNNEEG(nn.Module):
    """
    RNN-based EEG Denoising Network.

    Architecture:
    - Bidirectional LSTM layers
    - Attention mechanism for focusing on relevant temporal features
    - Fully connected output layers
    """

    def __init__(self, input_size: int = 512, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input projection
        self.input_fc = nn.Linear(1, hidden_size // 2)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size // 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        # Output layers
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor, eog: torch.Tensor = None,
                emg: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input noisy EEG of shape (batch, seq_len)
            eog: EOG reference signal (optional)
            emg: EMG reference signal (optional)

        Returns:
            Denoised signal
        """
        batch_size, seq_len = x.shape

        # Reshape for processing
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)

        # Input projection
        x = self.input_fc(x)

        # LSTM processing
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Self-attention
        att_weights = self.attention(lstm_out)
        context = lstm_out * att_weights  # Element-wise attention

        # Output
        out = self.output_fc(context)

        return out.squeeze(-1)


class ResNetEEG(nn.Module):
    """
    ResNet-based EEG Denoising Network.

    Architecture:
    - 1D ResNet blocks with skip connections
    - Global average pooling
    - Transposed convolutions for upsampling
    """

    def __init__(self, input_size: int = 512, base_channels: int = 64,
                 num_blocks: int = 4):
        super().__init__()
        self.input_size = input_size

        # Input convolution
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU()
        )

        # ResNet blocks
        self.res_blocks = nn.ModuleList()
        in_channels = base_channels
        for i in range(num_blocks):
            out_channels = base_channels * (2 ** min(i, 2))
            self.res_blocks.append(
                ResBlock1D(in_channels, out_channels,
                          stride=2 if i > 0 else 1)
            )
            in_channels = out_channels

        # Upsampling path
        self.upsample = nn.ModuleList()
        for i in range(num_blocks - 1, 0, -1):
            out_channels = base_channels * (2 ** min(i - 1, 2))
            self.upsample.append(
                nn.Sequential(
                    nn.ConvTranspose1d(in_channels, out_channels,
                                      kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU()
                )
            )
            in_channels = out_channels

        # Output convolution
        self.output_conv = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        original_size = x.shape[-1]

        # Input convolution
        h = self.input_conv(x)

        # Encoder path with skip connections
        skip_connections = [h]
        for res_block in self.res_blocks:
            h = res_block(h)
            skip_connections.append(h)

        # Decoder path
        for i, upsample in enumerate(self.upsample):
            h = upsample(h)
            # Add skip connection if sizes match
            skip_idx = len(skip_connections) - 2 - i
            if skip_idx >= 0:
                skip = skip_connections[skip_idx]
                if h.shape[-1] == skip.shape[-1]:
                    h = h + skip

        # Output
        out = self.output_conv(h)

        # Match output size to input
        if out.shape[-1] != original_size:
            out = F.interpolate(out, size=original_size, mode='linear', align_corners=False)

        return out.squeeze(1)


class ResBlock1D(nn.Module):
    """1D Residual Block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class SimpleCNN(nn.Module):
    """Simple CNN baseline for EEG denoising."""

    def __init__(self, input_size: int = 512, num_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, num_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(num_channels, num_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(num_channels, num_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(num_channels, 1, kernel_size=5, padding=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.net(x).squeeze(1)


# =============================================================================
# Traditional Filter Baselines
# =============================================================================

class WienerFilter:
    """Wiener filter for EEG denoising."""

    def __init__(self, wiener_size: int = 5):
        self.wiener_size = wiener_size

    def __call__(self, noisy_eeg: np.ndarray, eeg_stds: np.ndarray,
                 eeg_means: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Wiener filter."""
        num_samples = noisy_eeg.shape[0]
        filtered_eeg = np.zeros_like(noisy_eeg)

        for i in range(num_samples):
            try:
                noisy_sample = noisy_eeg[i].flatten()
                filtered = wiener(noisy_sample, self.wiener_size)
                filtered_eeg[i] = filtered * eeg_stds[i] + eeg_means[i]
            except Exception as e:
                logging.warning(f"Wiener filter failed for sample {i}: {e}")
                filtered_eeg[i] = noisy_eeg[i] * eeg_stds[i] + eeg_means[i]

        return filtered_eeg


class LMSFilter:
    """Least Mean Squares adaptive filter."""

    def __init__(self, mu: float = 0.001, filter_order: int = 32):
        self.mu = mu
        self.M = filter_order

    def __call__(self, noisy_eeg: np.ndarray, eog_normalized: np.ndarray,
                 emg_normalized: np.ndarray, eeg_stds: np.ndarray,
                 eeg_means: np.ndarray, **kwargs) -> np.ndarray:
        """Apply LMS filter."""
        eps = 1e-10
        filtered_output = np.zeros_like(noisy_eeg, dtype=np.float64)

        for ch in range(noisy_eeg.shape[0]):
            try:
                d = noisy_eeg[ch].astype(np.float64)
                x_eog = eog_normalized[ch].astype(np.float64)
                x_emg = emg_normalized[ch].astype(np.float64)

                w_eog = np.zeros(self.M)
                w_emg = np.zeros(self.M)

                for n in range(self.M, len(d)):
                    x_eog_window = x_eog[n - self.M:n]
                    x_emg_window = x_emg[n - self.M:n]

                    y_eog = np.dot(w_eog, x_eog_window)
                    y_emg = np.dot(w_emg, x_emg_window)

                    e = d[n] - (y_eog + y_emg)

                    power_eog = np.dot(x_eog_window, x_eog_window) + eps
                    power_emg = np.dot(x_emg_window, x_emg_window) + eps

                    w_eog = w_eog + 2 * (self.mu / power_eog) * e * x_eog_window
                    w_emg = w_emg + 2 * (self.mu / power_emg) * e * x_emg_window

                    w_eog = np.clip(w_eog, -1.0, 1.0)
                    w_emg = np.clip(w_emg, -1.0, 1.0)

                    filtered_output[ch, n] = d[n] - (y_eog + y_emg)

                filtered_output[ch, :self.M] = d[:self.M]

            except Exception as e:
                logging.warning(f"LMS filter failed for channel {ch}: {e}")
                filtered_output[ch] = noisy_eeg[ch]

        filtered_output = np.nan_to_num(filtered_output, nan=0.0)
        filtered_output = filtered_output * eeg_stds + eeg_means

        return filtered_output


class RLSFilter:
    """Recursive Least Squares filter."""

    def __init__(self, filter_order: int = 32, lambda_: float = 0.99,
                 delta: float = 0.01):
        self.M = filter_order
        self.lambda_ = lambda_
        self.delta = delta

    def __call__(self, noisy_eeg: np.ndarray, eog_normalized: np.ndarray,
                 emg_normalized: np.ndarray, eeg_stds: np.ndarray,
                 eeg_means: np.ndarray, **kwargs) -> np.ndarray:
        """Apply RLS filter."""
        filtered_output = np.zeros_like(noisy_eeg)

        for ch in range(noisy_eeg.shape[0]):
            d = noisy_eeg[ch]
            x_eog = eog_normalized[ch]
            x_emg = emg_normalized[ch]

            w = np.zeros(2 * self.M)
            P = self.delta * np.eye(2 * self.M)

            for n in range(self.M, len(d)):
                x_eog_window = x_eog[n - self.M:n]
                x_emg_window = x_emg[n - self.M:n]
                x = np.concatenate([x_eog_window, x_emg_window])

                k = P.dot(x) / (self.lambda_ + x.dot(P).dot(x))
                e = d[n] - w.dot(x)
                w = w + k * e
                P = (P - np.outer(k, x.dot(P))) / self.lambda_

                filtered_output[ch, n] = d[n] - w.dot(x)

            filtered_output[ch, :self.M] = d[:self.M]

        filtered_output = filtered_output * eeg_stds + eeg_means
        return filtered_output


class KalmanFilter:
    """Kalman filter for EEG denoising."""

    def __init__(self, process_noise: float = 0.001, measurement_noise: float = 0.1):
        self.Q = process_noise
        self.R = measurement_noise

    def __call__(self, noisy_eeg: np.ndarray, eeg_stds: np.ndarray,
                 eeg_means: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Kalman filter."""
        n_channels = noisy_eeg.shape[0]
        n_samples = noisy_eeg.shape[1]

        F = np.array([[1, 1], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.array([[self.Q, 0], [0, self.Q]])
        R = np.array([[self.R]])

        filtered_output = np.zeros_like(noisy_eeg)

        for ch in range(n_channels):
            x = np.array([[noisy_eeg[ch, 0]], [0]])
            P = np.eye(2)

            for t in range(n_samples):
                # Predict
                x_pred = F.dot(x)
                P_pred = F.dot(P).dot(F.T) + Q

                # Update
                z = np.array([[noisy_eeg[ch, t]]])
                y = z - H.dot(x_pred)
                S = H.dot(P_pred).dot(H.T) + R
                K = P_pred.dot(H.T).dot(np.linalg.inv(S))

                x = x_pred + K.dot(y)
                P = (np.eye(2) - K.dot(H)).dot(P_pred)

                filtered_output[ch, t] = x[0, 0]

        filtered_output = filtered_output * eeg_stds + eeg_means
        return filtered_output


# =============================================================================
# Training Functions for Deep Learning Models
# =============================================================================

def train_baseline_model(model: nn.Module, noisy_eeg: np.ndarray,
                        clean_eeg: np.ndarray, eog: np.ndarray = None,
                        emg: np.ndarray = None, epochs: int = 50,
                        batch_size: int = 32, lr: float = 0.001,
                        seed: int = 42) -> nn.Module:
    """
    Train a baseline deep learning model.

    Args:
        model: PyTorch model to train
        noisy_eeg: Noisy EEG data
        clean_eeg: Clean EEG targets
        eog: EOG reference (optional)
        emg: EMG reference (optional)
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        seed: Random seed

    Returns:
        Trained model
    """
    set_all_seeds(seed)

    # Convert to tensors
    noisy_tensor = torch.FloatTensor(noisy_eeg)
    clean_tensor = torch.FloatTensor(clean_eeg)

    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_batches = len(noisy_eeg) // batch_size

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            batch_noisy = noisy_tensor[start_idx:end_idx]
            batch_clean = clean_tensor[start_idx:end_idx]

            optimizer.zero_grad()

            # Forward pass (handle different model interfaces)
            if isinstance(model, RNNEEG) and eog is not None:
                batch_eog = torch.FloatTensor(eog[start_idx:end_idx]) if eog is not None else None
                batch_emg = torch.FloatTensor(emg[start_idx:end_idx]) if emg is not None else None
                output = model(batch_noisy, batch_eog, batch_emg)
            else:
                output = model(batch_noisy)

            loss = criterion(output, batch_clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if epoch % 10 == 0:
            avg_loss = total_loss / n_batches
            logging.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

    return model


def apply_baseline_model(model: nn.Module, noisy_eeg: np.ndarray,
                        eeg_stds: np.ndarray, eeg_means: np.ndarray,
                        eog: np.ndarray = None, emg: np.ndarray = None,
                        batch_size: int = 32) -> np.ndarray:
    """
    Apply a trained baseline model to denoise EEG.

    Args:
        model: Trained PyTorch model
        noisy_eeg: Noisy EEG data
        eeg_stds: Standard deviations for denormalization
        eeg_means: Means for denormalization
        eog: EOG reference (optional)
        emg: EMG reference (optional)
        batch_size: Batch size for inference

    Returns:
        Denoised EEG data
    """
    model.eval()
    noisy_tensor = torch.FloatTensor(noisy_eeg)

    with torch.no_grad():
        outputs = []
        for i in range(0, len(noisy_eeg), batch_size):
            batch = noisy_tensor[i:i + batch_size]

            if isinstance(model, RNNEEG) and eog is not None:
                batch_eog = torch.FloatTensor(eog[i:i + batch_size]) if eog is not None else None
                batch_emg = torch.FloatTensor(emg[i:i + batch_size]) if emg is not None else None
                out = model(batch, batch_eog, batch_emg)
            else:
                out = model(batch)

            outputs.append(out)

        filtered = torch.cat(outputs, dim=0).numpy()

    # Denormalize
    filtered = filtered * eeg_stds + eeg_means
    return filtered
