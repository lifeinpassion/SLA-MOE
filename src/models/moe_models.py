"""
Mixture of Experts (MoE) models for EEG denoising.

This module contains:
- RNNMoEFilter: Basic RNN-MoE filter
- ICAMoEFilter: ICA-enhanced MoE filter with self-learning pre-training
- ReservoirMoEFilter: MoE with reservoir computing enhancement
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import warnings
from typing import Optional, Tuple, Dict, List
from sklearn.decomposition import FastICA, PCA

from ..utils.seed_utils import set_all_seeds


# =============================================================================
# ICA Decomposer
# =============================================================================

class RobustICADecomposer:
    """ICA decomposer with robust handling and dynamic component management."""

    def __init__(self, n_components: int = 4, random_state: int = None,
                 max_iter: int = 800, tol: float = 1e-3,
                 algorithm: str = 'parallel', fun: str = 'logcosh'):
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.algorithm = algorithm
        self.fun = fun
        self.mixing_matrix = None
        self.unmixing_matrix = None
        self.fitted = False
        self.actual_components = n_components
        self.ica = None
        self.pca = None

    def fit(self, signals: np.ndarray):
        """Fit ICA with robust error handling."""
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        max_components = min(self.n_components, signals.shape[0], signals.shape[1])

        configs = [
            {'n_components': max_components, 'max_iter': self.max_iter,
             'tol': self.tol, 'algorithm': self.algorithm, 'fun': self.fun},
            {'n_components': max_components, 'max_iter': 500,
             'tol': 5e-3, 'algorithm': self.algorithm, 'fun': self.fun},
            {'n_components': max_components, 'max_iter': 300,
             'tol': 1e-2, 'algorithm': 'deflation', 'fun': self.fun},
            {'n_components': min(max_components, 3), 'max_iter': 200,
             'tol': 1e-2, 'algorithm': 'parallel', 'fun': 'logcosh'},
        ]

        for i, config in enumerate(configs):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    warnings.filterwarnings('ignore', category=RuntimeWarning)

                    self.ica = FastICA(random_state=self.random_state, **config)

                    signals_noisy = signals + np.random.RandomState(
                        self.random_state).randn(*signals.shape) * 1e-8

                    self.ica.fit(signals_noisy.T)
                    self.mixing_matrix = self.ica.mixing_
                    self.unmixing_matrix = self.ica.components_
                    self.fitted = True
                    self.actual_components = config['n_components']

                    if i == 0:
                        logging.debug(f"ICA fitted with {self.actual_components} components")
                    return

            except Exception as e:
                if i == len(configs) - 1:
                    logging.warning(f"ICA fitting failed, using PCA fallback: {e}")
                    self._use_pca_fallback(signals)
                continue

    def _use_pca_fallback(self, signals: np.ndarray):
        """Use PCA as fallback."""
        try:
            self.pca = PCA(n_components=min(self.n_components,
                                           signals.shape[0], signals.shape[1]),
                          random_state=self.random_state)
            self.pca.fit(signals.T)
            self.fitted = True
            self.actual_components = self.pca.n_components_
            logging.info(f"Using PCA with {self.actual_components} components")
        except:
            self.fitted = False
            self.actual_components = 0

    def decompose(self, signal: np.ndarray) -> Optional[np.ndarray]:
        """Decompose signal into components."""
        if not self.fitted:
            return None

        if torch.is_tensor(signal):
            signal_np = signal.detach().cpu().numpy()
        else:
            signal_np = signal

        if signal_np.ndim == 1:
            signal_np = signal_np.reshape(1, -1)

        try:
            if self.pca is not None:
                components = self.pca.transform(signal_np.T).T
            else:
                components = self.ica.transform(signal_np.T).T
            return components
        except:
            return None

    def reconstruct_component(self, signal, component_indices: List[int]):
        """Reconstruct signal using selected components."""
        if not self.fitted:
            return signal

        components = self.decompose(signal)
        if components is None:
            return signal

        valid_indices = [idx for idx in component_indices if idx < components.shape[0]]
        if not valid_indices:
            return signal

        selected_components = np.zeros_like(components)
        for idx in valid_indices:
            selected_components[idx] = components[idx]

        try:
            if self.pca is not None:
                reconstructed = self.pca.inverse_transform(selected_components.T).T
            else:
                reconstructed = self.ica.inverse_transform(selected_components.T).T

            if reconstructed.ndim > 1 and reconstructed.shape[0] == 1:
                reconstructed = reconstructed.squeeze(0)

            return torch.FloatTensor(reconstructed) if torch.is_tensor(signal) else reconstructed
        except:
            return signal

    def get_component_characteristics(self, component: np.ndarray) -> Dict:
        """Get characteristics of a component."""
        try:
            if len(component) < 2:
                return self._default_characteristics()

            comp_mean = np.mean(component)
            comp_std = np.std(component) + 1e-10
            normalized = (component - comp_mean) / comp_std
            skewness = np.mean(normalized ** 3)
            kurtosis = np.mean(normalized ** 4) - 3

            fft = np.fft.rfft(component)
            freqs = np.fft.rfftfreq(len(component))
            power = np.abs(fft) ** 2

            if len(power) > 1:
                dominant_freq_idx = np.argmax(power[1:]) + 1
                dominant_freq = freqs[dominant_freq_idx] if dominant_freq_idx < len(freqs) else 0
                total_power = np.sum(power) + 1e-10
                low_boundary = max(1, len(power) // 4)
                high_boundary = max(1, len(power) // 2)
                low_freq_power = np.sum(power[:low_boundary]) / total_power
                high_freq_power = np.sum(power[high_boundary:]) / total_power
            else:
                dominant_freq = 0
                low_freq_power = 0.5
                high_freq_power = 0.5

            return {
                'mean': comp_mean,
                'std': comp_std,
                'skewness': np.clip(skewness, -10, 10),
                'kurtosis': np.clip(kurtosis, -10, 10),
                'dominant_freq': dominant_freq,
                'low_freq_power': low_freq_power,
                'high_freq_power': high_freq_power
            }
        except:
            return self._default_characteristics()

    def _default_characteristics(self) -> Dict:
        return {
            'mean': 0.0, 'std': 1.0, 'skewness': 0.0, 'kurtosis': 0.0,
            'dominant_freq': 0.0, 'low_freq_power': 0.5, 'high_freq_power': 0.5
        }


# =============================================================================
# Expert Networks
# =============================================================================

class RNNExpert(nn.Module):
    """Basic RNN Expert for MoE."""

    def __init__(self, expert_id: int, hidden_size: int = 128,
                 input_features: int = 3):
        super().__init__()
        self.expert_id = expert_id
        self.hidden_size = hidden_size

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.lstm = nn.LSTM(64, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights based on expert role."""
        if self.expert_id == 0:
            nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        elif self.expert_id == 1:
            nn.init.uniform_(self.fc.weight, -0.5, 0.5)
        elif self.expert_id == 2:
            nn.init.uniform_(self.fc.weight, -0.2, 0.2)
        else:
            nn.init.uniform_(self.fc.weight, -0.3, 0.3)

    def forward(self, x: torch.Tensor, eog: torch.Tensor,
                emg: torch.Tensor) -> torch.Tensor:
        features = torch.stack([x, eog, emg], dim=-1)
        feature_out = self.feature_extractor(features)
        lstm_out, _ = self.lstm(feature_out)
        output = self.fc(lstm_out)
        return output.squeeze(-1)


class ICAEnhancedExpert(nn.Module):
    """Expert network with ICA feature enhancement."""

    def __init__(self, expert_id: int, hidden_size: int = 128,
                 input_features: int = 7):
        super().__init__()
        self.expert_id = expert_id
        self.input_features = input_features

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64)
        )

        self.lstm = nn.LSTM(64, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.feature_extractor:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor, eog: torch.Tensor, emg: torch.Tensor,
                ica_features: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = x.shape
        basic_features = torch.stack([x, eog, emg], dim=-1)

        if ica_features is not None and ica_features.shape[-1] > 0:
            all_features = torch.cat([basic_features, ica_features], dim=-1)
        else:
            # Pad to expected input size
            padding = torch.zeros(batch_size, seq_len,
                                 self.input_features - 3, device=x.device)
            all_features = torch.cat([basic_features, padding], dim=-1)

        features = self.feature_extractor(all_features)
        lstm_out, _ = self.lstm(features)
        output = self.fc(lstm_out)
        return output.squeeze(-1)


# =============================================================================
# MoE Models
# =============================================================================

class RNNMoEFilter(nn.Module):
    """
    Basic RNN-based Mixture of Experts filter for EEG denoising.
    """

    def __init__(self, num_experts: int = 4, hidden_size: int = 128,
                 top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            RNNExpert(i, hidden_size) for i in range(num_experts)
        ])

        self.gate_lstm = nn.LSTM(3, 64, batch_first=True)
        self.gate_network = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts)
        )

        self.gate_probs = None

    def forward(self, x: torch.Tensor, eog: torch.Tensor,
                emg: torch.Tensor) -> torch.Tensor:
        gate_input = torch.stack([x, eog, emg], dim=-1)
        gate_lstm_out, _ = self.gate_lstm(gate_input)
        gate_logits = self.gate_network(gate_lstm_out)
        gate_probs = F.softmax(gate_logits, dim=2)

        # Sparse gating
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=2)
        mask = torch.zeros_like(gate_probs)
        mask.scatter_(2, top_k_indices, top_k_probs)
        gate_probs = mask / mask.sum(dim=2, keepdim=True).clamp(min=1e-10)

        self.gate_probs = gate_probs

        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x, eog, emg)
            expert_outputs.append(expert_out)

        stacked_outputs = torch.stack(expert_outputs, dim=2)
        combined_output = torch.sum(stacked_outputs * gate_probs, dim=2)

        return combined_output


class ICAMoEFilter(nn.Module):
    """
    ICA-enhanced Mixture of Experts filter with self-learning pre-training.

    This is the main IAC-initialization-enhanced MoE model.
    """

    def __init__(self, num_experts: int = 4, hidden_size: int = 128,
                 n_ica_components: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.n_ica_components = n_ica_components
        self.hidden_size = hidden_size

        self.ica_decomposer = None
        input_features = 3 + n_ica_components

        self.experts = nn.ModuleList([
            ICAEnhancedExpert(i, hidden_size, input_features)
            for i in range(num_experts)
        ])

        self.gate_lstm = nn.LSTM(input_features, 64, batch_first=True)
        self.gate_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_experts)
        )

        self.gate_probs = None
        self.component_characteristics = []

    def set_ica_decomposer(self, ica_decomposer: RobustICADecomposer):
        """Set the ICA decomposer after fitting."""
        self.ica_decomposer = ica_decomposer

    def compute_ica_features(self, signals: torch.Tensor) -> torch.Tensor:
        """Compute ICA features for a batch of signals."""
        batch_size, seq_len = signals.shape

        if self.ica_decomposer is None or not self.ica_decomposer.fitted:
            return torch.zeros(batch_size, seq_len, self.n_ica_components)

        n_components = min(self.ica_decomposer.actual_components, self.n_ica_components)
        ica_features = torch.zeros(batch_size, seq_len, self.n_ica_components)

        for i in range(batch_size):
            signal = signals[i].cpu().numpy()
            window_size = min(64, seq_len)
            stride = max(8, window_size // 8)

            weights = np.zeros((seq_len, n_components))
            counts = np.zeros(seq_len)

            for j in range(0, max(1, seq_len - window_size + 1), stride):
                end_j = min(j + window_size, seq_len)
                window = signal[j:end_j] * np.hanning(end_j - j)

                if len(window) >= 4:
                    components = self.ica_decomposer.decompose(window)
                    if components is not None:
                        for k in range(j, end_j):
                            local_idx = k - j
                            w = 1.0 - abs(2.0 * local_idx / len(window) - 1.0)
                            for comp_idx in range(min(n_components, components.shape[0])):
                                if local_idx < components.shape[1]:
                                    weights[k, comp_idx] += w * components[comp_idx, local_idx]
                                    counts[k] += w

            for k in range(seq_len):
                if counts[k] > 0:
                    ica_features[i, k, :n_components] = torch.FloatTensor(
                        weights[k, :] / counts[k])

        feature_norm = torch.norm(ica_features, p=2, dim=-1, keepdim=True)
        ica_features = ica_features / (feature_norm + 1e-10)

        return ica_features

    def forward(self, x: torch.Tensor, eog: torch.Tensor,
                emg: torch.Tensor) -> torch.Tensor:
        ica_features = self.compute_ica_features(x)
        basic_features = torch.stack([x, eog, emg], dim=-1)
        gate_input = torch.cat([basic_features, ica_features], dim=-1)

        gate_hidden, _ = self.gate_lstm(gate_input)
        gate_logits = self.gate_fc(gate_hidden)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Sparse gating
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        mask = torch.zeros_like(gate_probs)
        mask.scatter_(-1, top_k_indices, top_k_probs)
        gate_probs = mask / (mask.sum(dim=-1, keepdim=True) + 1e-10)

        self.gate_probs = gate_probs

        expert_outputs = []
        for expert in self.experts:
            output = expert(x, eog, emg, ica_features)
            expert_outputs.append(output)

        stacked = torch.stack(expert_outputs, dim=2)
        combined = torch.sum(stacked * gate_probs, dim=2)

        return combined


class EchoStateReservoir(nn.Module):
    """Echo State Network Reservoir Computing layer."""

    def __init__(self, input_size: int, reservoir_size: int = 500,
                 spectral_radius: float = 0.95, sparsity: float = 0.1,
                 input_scaling: float = 1.0, leaking_rate: float = 0.3):
        super().__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate

        self._initialize_reservoir()
        self.readout = nn.Linear(reservoir_size + input_size, 1)
        self.ridge_param = 1e-4

    def _initialize_reservoir(self):
        W_in = (torch.randn(self.reservoir_size, self.input_size) - 0.5) * 2
        self.register_buffer('W_in', W_in * self.input_scaling)

        W = torch.randn(self.reservoir_size, self.reservoir_size)
        mask = torch.rand(self.reservoir_size, self.reservoir_size) < self.sparsity
        W = W * mask.float()

        eigenvalues = torch.linalg.eigvals(W).abs()
        max_eigenvalue = eigenvalues.max()
        if max_eigenvalue > 0:
            W = W * (self.spectral_radius / max_eigenvalue)

        self.register_buffer('W_reservoir', W)
        bias = (torch.randn(self.reservoir_size) - 0.5) * 2
        self.register_buffer('reservoir_bias', bias * 0.1)

    def reservoir_forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        states = []
        h = torch.zeros(batch_size, self.reservoir_size, device=device)

        for t in range(seq_len):
            input_contribution = torch.mm(x[:, t], self.W_in.T)
            recurrent_contribution = torch.mm(h, self.W_reservoir.T)
            pre_activation = input_contribution + recurrent_contribution + self.reservoir_bias
            new_h = torch.tanh(pre_activation)
            h = (1 - self.leaking_rate) * h + self.leaking_rate * new_h
            states.append(h)

        return torch.stack(states, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        reservoir_states = self.reservoir_forward(x)
        combined = torch.cat([x, reservoir_states], dim=-1)
        output = self.readout(combined)
        return output.squeeze(-1)


class ReservoirMoEFilter(nn.Module):
    """
    MoE filter with Reservoir Computing enhancement.

    Pipeline: Input -> ICA features -> MoE -> Reservoir -> Output
    """

    def __init__(self, num_experts: int = 4, hidden_size: int = 128,
                 n_ica_components: int = 4, reservoir_size: int = 500):
        super().__init__()

        self.moe = ICAMoEFilter(num_experts, hidden_size, n_ica_components)
        self.reservoir = EchoStateReservoir(
            input_size=1,
            reservoir_size=reservoir_size,
            spectral_radius=0.9,
            sparsity=0.1,
            input_scaling=0.5,
            leaking_rate=0.3
        )

        self.refinement = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )

    def set_ica_decomposer(self, ica_decomposer: RobustICADecomposer):
        self.moe.set_ica_decomposer(ica_decomposer)

    def forward(self, x: torch.Tensor, eog: torch.Tensor,
                emg: torch.Tensor) -> torch.Tensor:
        moe_output = self.moe(x, eog, emg)
        reservoir_output = self.reservoir(moe_output)

        batch_size, seq_len = reservoir_output.shape
        reservoir_reshaped = reservoir_output.reshape(batch_size * seq_len, 1)
        refined = self.refinement(reservoir_reshaped)
        output = refined.reshape(batch_size, seq_len)

        return output


# =============================================================================
# Training Functions
# =============================================================================

class ICASelfLearningPretrainer:
    """
    ICA-based self-learning pre-training for expert specialization.
    """

    def __init__(self, model: ICAMoEFilter, noisy_eeg: torch.Tensor,
                 eog: torch.Tensor, emg: torch.Tensor, clean_eeg: torch.Tensor,
                 ica_decomposer: RobustICADecomposer, seed: int = 42):
        self.model = model
        self.noisy_eeg = noisy_eeg
        self.eog = eog
        self.emg = emg
        self.clean_eeg = clean_eeg
        self.ica_decomposer = ica_decomposer
        self.seed = seed
        self.component_characteristics = []

        self._analyze_components()

    def _analyze_components(self):
        """Analyze ICA component characteristics."""
        if self.ica_decomposer.fitted and self.ica_decomposer.actual_components > 0:
            sample_components = self.ica_decomposer.decompose(
                self.noisy_eeg[0].cpu().numpy())
            if sample_components is not None:
                for i in range(sample_components.shape[0]):
                    chars = self.ica_decomposer.get_component_characteristics(
                        sample_components[i])
                    self.component_characteristics.append(chars)

    def create_targets(self, batch_noisy: torch.Tensor, batch_clean: torch.Tensor,
                      expert_id: int) -> torch.Tensor:
        """Create expert-specific targets based on ICA analysis."""
        batch_size = batch_noisy.shape[0]
        targets = []

        for i in range(batch_size):
            noisy_signal = batch_noisy[i]
            clean_signal = batch_clean[i]

            if not self.ica_decomposer.fitted or not self.component_characteristics:
                alpha = 0.9 - 0.1 * expert_id
                target = alpha * clean_signal + (1 - alpha) * noisy_signal
            else:
                if expert_id == 0:
                    low_freq_comps = [idx for idx, c in enumerate(
                        self.component_characteristics) if c['low_freq_power'] > 0.6]
                    if low_freq_comps:
                        recon = self.ica_decomposer.reconstruct_component(
                            noisy_signal, low_freq_comps[:2])
                        target = 0.7 * clean_signal + 0.3 * recon
                    else:
                        target = 0.8 * clean_signal + 0.2 * noisy_signal

                elif expert_id == 1:
                    high_freq_comps = [idx for idx, c in enumerate(
                        self.component_characteristics) if c['high_freq_power'] > 0.4]
                    if high_freq_comps:
                        recon = self.ica_decomposer.reconstruct_component(
                            noisy_signal, high_freq_comps[:2])
                        target = 0.75 * clean_signal + 0.25 * recon
                    else:
                        target = 0.85 * clean_signal + 0.15 * noisy_signal

                elif expert_id == 2:
                    artifact_comps = [idx for idx, c in enumerate(
                        self.component_characteristics)
                        if abs(c['kurtosis']) > 1 or abs(c['skewness']) > 1]
                    if artifact_comps:
                        artifacts = self.ica_decomposer.reconstruct_component(
                            noisy_signal, artifact_comps[:1])
                        target = clean_signal - 0.15 * (artifacts - torch.mean(artifacts))
                    else:
                        target = 0.9 * clean_signal + 0.1 * noisy_signal

                else:
                    all_comps = list(range(len(self.component_characteristics)))[:3]
                    if all_comps:
                        recon = self.ica_decomposer.reconstruct_component(
                            noisy_signal, all_comps)
                        target = 0.6 * clean_signal + 0.4 * recon
                    else:
                        target = 0.8 * clean_signal + 0.2 * noisy_signal

            targets.append(target)

        return torch.stack(targets)

    def pretrain(self, epochs: int = 5, batch_size: int = 32):
        """Pretrain experts with ICA-based self-learning."""
        logging.info(f"Starting ICA-based pretraining with "
                    f"{self.ica_decomposer.actual_components} components...")

        optimizers = [torch.optim.Adam(expert.parameters(), lr=0.003)
                     for expert in self.model.experts]
        criterion = nn.SmoothL1Loss()

        n_batches = max(1, len(self.noisy_eeg) // batch_size)

        for epoch in range(epochs):
            total_losses = [0.0] * len(self.model.experts)

            torch.manual_seed(self.seed + epoch)
            indices = torch.randperm(len(self.noisy_eeg))

            for i in range(n_batches):
                batch_idx = indices[i * batch_size:min((i + 1) * batch_size, len(indices))]

                batch_noisy = self.noisy_eeg[batch_idx]
                batch_eog = self.eog[batch_idx]
                batch_emg = self.emg[batch_idx]
                batch_clean = self.clean_eeg[batch_idx]

                ica_features = self.model.compute_ica_features(batch_noisy)

                for expert_id, (expert, optimizer) in enumerate(
                        zip(self.model.experts, optimizers)):
                    optimizer.zero_grad()

                    target = self.create_targets(batch_noisy, batch_clean, expert_id)
                    output = expert(batch_noisy, batch_eog, batch_emg, ica_features)

                    loss = criterion(output, target)

                    if expert_id > 0:
                        for prev_id in range(expert_id):
                            prev_out = self.model.experts[prev_id](
                                batch_noisy, batch_eog, batch_emg, ica_features).detach()
                            diversity_penalty = F.cosine_similarity(
                                output.view(-1), prev_out.view(-1), dim=0)
                            loss = loss + 0.02 * torch.abs(diversity_penalty)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(expert.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_losses[expert_id] += loss.item()

            if epoch % 2 == 0:
                avg_losses = [l / n_batches for l in total_losses]
                logging.debug(f"Pretrain Epoch {epoch}: {avg_losses}")


def compute_load_balancing_loss(gate_probs: torch.Tensor) -> torch.Tensor:
    """Compute load balancing loss for MoE training."""
    router_probs = gate_probs.mean(dim=[0, 1])
    router_load = gate_probs.sum(dim=[0, 1])
    num_experts = router_probs.size(0)
    loss = torch.sum(router_probs * router_load) * (num_experts ** 2)
    return loss


def train_ica_moe(model: ICAMoEFilter, noisy_eeg: np.ndarray,
                  clean_eeg: np.ndarray, eog: np.ndarray, emg: np.ndarray,
                  eeg_stds: np.ndarray, eeg_means: np.ndarray,
                  epochs: int = 10, pretrain_epochs: int = 5,
                  batch_size: int = 32, seed: int = 42) -> np.ndarray:
    """
    Train ICA-MoE model and return denoised output.

    Args:
        model: ICAMoEFilter model
        noisy_eeg: Noisy EEG data
        clean_eeg: Clean EEG targets
        eog: EOG reference
        emg: EMG reference
        eeg_stds: Standard deviations for denormalization
        eeg_means: Means for denormalization
        epochs: Main training epochs
        pretrain_epochs: Pre-training epochs
        batch_size: Batch size
        seed: Random seed

    Returns:
        Denoised EEG data
    """
    set_all_seeds(seed)

    noisy_tensor = torch.FloatTensor(noisy_eeg)
    clean_tensor = torch.FloatTensor(clean_eeg)
    eog_tensor = torch.FloatTensor(eog)
    emg_tensor = torch.FloatTensor(emg)

    # Initialize and fit ICA decomposer
    ica_decomposer = RobustICADecomposer(
        n_components=model.n_ica_components,
        random_state=seed,
        max_iter=800,
        tol=1e-3
    )

    n_samples_for_fit = min(500, len(noisy_eeg))
    torch.manual_seed(seed)
    sample_indices = torch.randperm(len(noisy_eeg))[:n_samples_for_fit].numpy()
    ica_decomposer.fit(noisy_eeg[sample_indices])

    model.set_ica_decomposer(ica_decomposer)

    # Pre-training
    pretrainer = ICASelfLearningPretrainer(
        model, noisy_tensor, eog_tensor, emg_tensor, clean_tensor,
        ica_decomposer, seed
    )
    pretrainer.pretrain(epochs=pretrain_epochs, batch_size=batch_size)

    # Main training
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_batches = max(1, len(noisy_eeg) // batch_size)

    logging.info("Starting main training...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(noisy_eeg))

            batch_noisy = noisy_tensor[start_idx:end_idx]
            batch_eog = eog_tensor[start_idx:end_idx]
            batch_emg = emg_tensor[start_idx:end_idx]
            batch_target = clean_tensor[start_idx:end_idx]

            optimizer.zero_grad()
            output = model(batch_noisy, batch_eog, batch_emg)
            loss = criterion(output, batch_target)

            if model.gate_probs is not None:
                lb_loss = compute_load_balancing_loss(model.gate_probs)
                loss = loss + 0.01 * lb_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if epoch % 2 == 0:
            avg_loss = total_loss / n_batches
            logging.debug(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

    # Apply model
    model.eval()
    with torch.no_grad():
        filtered = model(noisy_tensor, eog_tensor, emg_tensor)

    filtered = filtered.numpy() * eeg_stds + eeg_means
    return filtered
