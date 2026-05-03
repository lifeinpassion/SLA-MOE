"""
Original ICA-MoE model from initial_seeded.py

This is the best-performing model for the paper, preserved as the original implementation
with minor modifications for parameterized seed and noise type support.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import FastICA
import logging
import random
import warnings
from typing import Optional, Tuple, Dict

from ..utils.seed_utils import set_all_seeds


def apply_rnn_moe_filter_ica(noisy_eeg: np.ndarray,
                              eeg_normalized: np.ndarray,
                              eog_normalized: np.ndarray,
                              emg_normalized: np.ndarray,
                              eeg_stds: np.ndarray,
                              eeg_means: np.ndarray,
                              seed: int = 40,
                              device=None,
                              # ---- ablation knobs (JBHI revision) ----
                              num_experts: int = 4,
                              top_k: int = 2,
                              hidden_size: int = 128,
                              pretrain_epochs: int = 5,
                              main_epochs: int = 10,
                              lb_coef: float = 0.01,
                              lambda_independence: float = 0.1,
                              use_ica_features_in_gate: bool = True,
                              learning_rate: float = 1e-3,
                              ) -> np.ndarray:
    """
    Apply Classical Recurrent Neural Network with Mixture of Experts (RNN-MoE) filter to EEG data.
    This version uses ICA-based self-learning pre-training for expert specialization.

    GPU PORT (JBHI revision):
      * Auto-detects device (cuda > mps > cpu); honors caller's `device=` if given.
      * Model and all tensors live on `device` for the entire training+inference loop.
      * ICA features are pre-computed ONCE for the full dataset (replaces per-batch
        sklearn FastICA recompute, which previously dominated training time on
        both M1 MPS and Colab T4).

    Ablation knobs (JBHI revision):
      All of num_experts/top_k/pretrain_epochs/lb_coef/lambda_independence/
      use_ica_features_in_gate are exposed as kwargs so the ablation runner
      can vary them per variant. Defaults reproduce the headline configuration.

    Args:
        noisy_eeg: Noisy EEG data.
        eeg_normalized: Normalized EEG data.
        eog_normalized: Normalized EOG data.
        emg_normalized: Normalized EMG data.
        eeg_stds: Standard deviations for denormalization.
        eeg_means: Means for denormalization.
        seed: Random seed for reproducibility (default: 40)
        device: torch.device or string ('cuda'|'mps'|'cpu'); None auto-detects.
        num_experts, top_k, hidden_size, pretrain_epochs, main_epochs,
        lb_coef, lambda_independence, use_ica_features_in_gate, learning_rate:
            ablation knobs; see paper.

    Returns:
        Filtered EEG data (numpy on CPU, denormalized).
    """
    # Set all random seeds for reproducibility
    OPTIMAL_SEED = seed
    random.seed(OPTIMAL_SEED)
    np.random.seed(OPTIMAL_SEED)
    torch.manual_seed(OPTIMAL_SEED)
    torch.cuda.manual_seed(OPTIMAL_SEED)
    torch.cuda.manual_seed_all(OPTIMAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Device detection (cuda > mps > cpu)
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    logging.info(f"SLA-MoE applier using device: {device}  "
                 f"[E={num_experts}, k={top_k}, pretrain={pretrain_epochs}, "
                 f"lb={lb_coef}, ind={lambda_independence}, "
                 f"ica_gate={use_ica_features_in_gate}]")

    sequence_length = noisy_eeg.shape[1]

    # Convert inputs to tensors and move to device
    noisy_eeg_tensor = torch.FloatTensor(noisy_eeg).to(device)
    eog_tensor = torch.FloatTensor(eog_normalized).to(device)
    emg_tensor = torch.FloatTensor(emg_normalized).to(device)
    clean_tensor_full = torch.FloatTensor(eeg_normalized).to(device)

    class ICADecomposer:
        """Helper class for ICA decomposition operations."""
        def __init__(self, n_components=4):
            self.n_components = n_components
            self.ica = FastICA(
                n_components=n_components,
                random_state=OPTIMAL_SEED,
                max_iter=800,
                tol=1e-3,
                algorithm='parallel',
                fun='logcosh'
            )
            self.mixing_matrix = None
            self.unmixing_matrix = None
            self.fitted = False

        def fit(self, signals):
            if signals.ndim == 1:
                signals = signals.reshape(1, -1)

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    signals_with_noise = signals + np.random.RandomState(OPTIMAL_SEED).randn(*signals.shape) * 1e-8
                    self.ica.fit(signals_with_noise.T)
                    self.mixing_matrix = self.ica.mixing_
                    self.unmixing_matrix = self.ica.components_
                    self.fitted = True
                    logging.info(f"ICA fitted successfully with {self.n_components} components")
            except Exception as e:
                logging.warning(f"ICA fitting failed: {e}. Using PCA-based initialization instead.")
                self.fitted = False

        def decompose(self, signal):
            if not self.fitted:
                return None

            if torch.is_tensor(signal):
                signal_np = signal.detach().cpu().numpy()
            else:
                signal_np = signal

            if signal_np.ndim == 1:
                signal_np = signal_np.reshape(1, -1)

            try:
                components = self.ica.transform(signal_np.T).T
                return components
            except:
                return None

        def reconstruct_component(self, signal, component_indices):
            if not self.fitted:
                return signal

            components = self.decompose(signal)
            if components is None:
                return signal

            selected_components = np.zeros_like(components)
            for idx in component_indices:
                if idx < components.shape[0]:
                    selected_components[idx] = components[idx]

            reconstructed = self.ica.inverse_transform(selected_components.T).T

            if reconstructed.ndim > 1 and reconstructed.shape[0] == 1:
                reconstructed = reconstructed.squeeze(0)

            return torch.FloatTensor(reconstructed) if torch.is_tensor(signal) else reconstructed

        def get_component_characteristics(self, component):
            fft = np.fft.rfft(component)
            freqs = np.fft.rfftfreq(len(component))
            power = np.abs(fft) ** 2

            dominant_freq_idx = np.argmax(power[1:]) + 1
            dominant_freq = freqs[dominant_freq_idx] if dominant_freq_idx < len(freqs) else 0

            comp_std = np.std(component)
            if comp_std > 1e-10:
                normalized = (component - np.mean(component)) / comp_std
                skewness = np.mean(normalized ** 3)
                kurtosis = np.mean(normalized ** 4) - 3
            else:
                skewness = 0
                kurtosis = 0

            stats = {
                'mean': np.mean(component),
                'std': comp_std,
                'skewness': np.clip(skewness, -10, 10),
                'kurtosis': np.clip(kurtosis, -10, 10),
                'dominant_freq': dominant_freq,
                'low_freq_power': np.sum(power[:len(power)//4]) / (np.sum(power) + 1e-10),
                'high_freq_power': np.sum(power[len(power)//2:]) / (np.sum(power) + 1e-10)
            }

            return stats

    class RNNExpert(nn.Module):
        def __init__(self, expert_id):
            super().__init__()
            self.expert_id = expert_id

            self.feature_extractor = nn.Sequential(
                nn.Linear(7, 64),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

            self.lstm = nn.LSTM(64, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

            self._initialize_random()

        def _initialize_random(self):
            nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.fc.bias, 0.0)

            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param, gain=0.1)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)

        def forward(self, x, eog, emg, ica_features):
            batch_size, seq_len = x.shape
            basic_features = torch.stack([x, eog, emg], dim=-1)
            all_features = torch.cat([basic_features, ica_features], dim=-1)
            feature_out = self.feature_extractor(all_features)
            lstm_out, _ = self.lstm(feature_out)
            output = self.fc(lstm_out)
            return output.squeeze(-1)

    class ICASelfLearningPretrainer:
        def __init__(self, experts, noisy_eeg, eog, emg, clean_eeg):
            self.experts = experts
            self.noisy_eeg = noisy_eeg
            self.eog = eog
            self.emg = emg
            self.clean_eeg = clean_eeg
            self.ica_decomposer = ICADecomposer(n_components=4)
            self.component_characteristics = []
            self.ica_features_cache = None   # (N, T, n_components) on `device`

            self._fit_ica()
            self._precompute_ica_features_for_dataset()

        def _precompute_ica_features_for_dataset(self):
            """
            One-time computation of ICA features for ALL noisy_eeg segments.
            Runs the windowed CPU FastICA pipeline once, builds a (N, T, n_components)
            tensor, and moves it to `device`. Both the pretrainer and the main
            RNNMoEFilter look up batches from this cache via indexing, eliminating
            the per-batch FastICA recompute that dominated training time.
            """
            if not self.ica_decomposer.fitted:
                logging.warning("ICA not fitted; cache will be all zeros.")
                self.ica_features_cache = torch.zeros(
                    self.noisy_eeg.shape[0],
                    self.noisy_eeg.shape[1],
                    self.ica_decomposer.n_components,
                    device=device,
                )
                return

            N, T = self.noisy_eeg.shape
            n_comp = self.ica_decomposer.n_components
            window_size = min(64, T)
            stride = 8
            cache_np = np.zeros((N, T, n_comp), dtype=np.float32)

            # Move noisy_eeg to CPU once for the windowed numpy ops
            data_cpu = self.noisy_eeg.detach().cpu().numpy()

            logging.info(f"Pre-computing ICA features for {N} segments (one-time)...")
            for i in range(N):
                signal = data_cpu[i]
                weights = np.zeros((T, n_comp), dtype=np.float32)
                counts = np.zeros(T, dtype=np.float32)

                for j in range(0, T - window_size + 1, stride):
                    window = signal[j:j + window_size] * np.hanning(window_size)
                    components = self.ica_decomposer.decompose(window)
                    if components is None:
                        continue
                    for comp_idx in range(min(n_comp, components.shape[0])):
                        comp = components[comp_idx]
                        comp_std = np.std(comp) + 1e-10
                        comp_normalized = comp / comp_std
                        for k in range(j, min(j + window_size, T)):
                            local_idx = int((k - j) * len(comp) / window_size)
                            weight = 1.0 - abs(2.0 * (k - j) / window_size - 1.0)
                            weights[k, comp_idx] += weight * comp_normalized[min(local_idx, len(comp) - 1)]
                            counts[k] += weight

                nz = counts > 0
                if nz.any():
                    cache_np[i, nz] = weights[nz] / counts[nz, None]

            cache_t = torch.from_numpy(cache_np).to(device)
            cache_t = F.normalize(cache_t, p=2, dim=-1)
            self.ica_features_cache = cache_t
            # Also share with the decomposer so RNNMoEFilter can use it
            self.ica_decomposer.features_cache = cache_t
            logging.info(f"ICA features cache built, shape={tuple(cache_t.shape)}, device={cache_t.device}")

        def _fit_ica(self):
            n_samples = min(500, len(self.noisy_eeg))
            torch.manual_seed(OPTIMAL_SEED)
            sample_indices = torch.randperm(len(self.noisy_eeg))[:n_samples].numpy()
            sample_data = self.noisy_eeg[sample_indices].cpu().numpy()

            self.ica_decomposer.fit(sample_data)

            if self.ica_decomposer.fitted:
                sample_components = self.ica_decomposer.decompose(sample_data[0])
                if sample_components is not None:
                    for i in range(sample_components.shape[0]):
                        characteristics = self.ica_decomposer.get_component_characteristics(sample_components[i])
                        self.component_characteristics.append(characteristics)
                        logging.info(f"ICA Component {i}: dominant_freq={characteristics['dominant_freq']:.3f}, "
                                   f"low_freq_power={characteristics['low_freq_power']:.2%}, "
                                   f"high_freq_power={characteristics['high_freq_power']:.2%}")

        def compute_ica_features(self, signals, batch_indices=None):
            """
            With the cache, this is a fast tensor lookup. If `batch_indices`
            is provided we slice the cache directly. The legacy slow path
            (no cache, recompute per batch) is preserved as a fallback.
            """
            if batch_indices is not None and self.ica_features_cache is not None:
                if torch.is_tensor(batch_indices):
                    batch_indices = batch_indices.detach().cpu().numpy()
                return self.ica_features_cache[batch_indices]

            # ---- legacy fallback (slow): compute on the fly ----
            batch_size, seq_len = signals.shape
            n_components = self.ica_decomposer.n_components
            ica_features = torch.zeros(batch_size, seq_len, n_components, device=signals.device)
            if not self.ica_decomposer.fitted:
                return ica_features

            window_size = min(64, seq_len)
            stride = 8
            for i in range(batch_size):
                signal_np = signals[i].detach().cpu().numpy()
                weights = np.zeros((seq_len, n_components), dtype=np.float32)
                counts = np.zeros(seq_len, dtype=np.float32)
                for j in range(0, seq_len - window_size + 1, stride):
                    window = signal_np[j:j + window_size] * np.hanning(window_size)
                    components = self.ica_decomposer.decompose(window)
                    if components is None:
                        continue
                    for comp_idx in range(min(n_components, components.shape[0])):
                        comp = components[comp_idx]
                        comp_std = np.std(comp) + 1e-10
                        comp_normalized = comp / comp_std
                        for k in range(j, min(j + window_size, seq_len)):
                            local_idx = int((k - j) * len(comp) / window_size)
                            weight = 1.0 - abs(2.0 * (k - j) / window_size - 1.0)
                            weights[k, comp_idx] += weight * comp_normalized[min(local_idx, len(comp) - 1)]
                            counts[k] += weight
                nz = counts > 0
                if nz.any():
                    block = weights[nz] / counts[nz, None]
                    ica_features[i, np.where(nz)[0], :] = torch.from_numpy(block).to(signals.device)
            ica_features = F.normalize(ica_features, p=2, dim=-1)
            return ica_features

        def create_ica_based_targets(self, batch_noisy, batch_clean, expert_id):
            batch_size, seq_len = batch_noisy.shape
            targets = []

            for i in range(batch_size):
                noisy_signal = batch_noisy[i]
                clean_signal = batch_clean[i]

                if not self.ica_decomposer.fitted:
                    noise_estimate = noisy_signal - clean_signal
                    target = clean_signal + 0.1 * torch.randn_like(clean_signal) * torch.std(noise_estimate)
                else:
                    if expert_id == 0:
                        low_freq_components = []
                        for idx, chars in enumerate(self.component_characteristics):
                            if chars['low_freq_power'] > 0.5:
                                low_freq_components.append(idx)

                        if low_freq_components:
                            reconstructed = self.ica_decomposer.reconstruct_component(noisy_signal, low_freq_components)
                        else:
                            reconstructed = self.ica_decomposer.reconstruct_component(noisy_signal, [0])

                        target = 0.7 * clean_signal + 0.3 * reconstructed

                    elif expert_id == 1:
                        high_freq_components = []
                        for idx, chars in enumerate(self.component_characteristics):
                            if chars['high_freq_power'] > 0.5:
                                high_freq_components.append(idx)

                        if high_freq_components:
                            reconstructed = self.ica_decomposer.reconstruct_component(noisy_signal, high_freq_components)
                        else:
                            reconstructed = self.ica_decomposer.reconstruct_component(noisy_signal, [1, 2])

                        high_freq_artifacts = noisy_signal - reconstructed
                        target = clean_signal - 0.3 * high_freq_artifacts

                    elif expert_id == 2:
                        non_gaussian_components = []
                        for idx, chars in enumerate(self.component_characteristics):
                            if abs(chars['kurtosis']) > 1 or abs(chars['skewness']) > 1:
                                non_gaussian_components.append(idx)

                        if non_gaussian_components:
                            artifact_estimate = self.ica_decomposer.reconstruct_component(noisy_signal, non_gaussian_components)
                            target = clean_signal - 0.2 * (artifact_estimate - torch.mean(artifact_estimate))
                        else:
                            reconstructed = self.ica_decomposer.reconstruct_component(noisy_signal, [1, 2])
                            target = 0.5 * clean_signal + 0.5 * reconstructed

                    else:
                        all_components = []
                        weights = []

                        for idx, chars in enumerate(self.component_characteristics):
                            all_components.append(idx)
                            cleanness = 1.0 / (1 + abs(chars['kurtosis']) + abs(chars['skewness']))
                            weights.append(cleanness)

                        weights = np.array(weights) / (np.sum(weights) + 1e-10)

                        reconstructed = torch.zeros_like(noisy_signal)
                        for comp_idx, weight in zip(all_components, weights):
                            comp_reconstructed = self.ica_decomposer.reconstruct_component(noisy_signal, [comp_idx])
                            reconstructed += weight * comp_reconstructed

                        target = 0.8 * clean_signal + 0.2 * reconstructed

                targets.append(target)

            return torch.stack(targets)

        def pretrain(self, epochs=5, batch_size=32):
            logging.info("Starting ICA-based self-learning pre-training phase...")

            optimizers = [torch.optim.Adam(expert.parameters(), lr=0.005)
                         for expert in self.experts]
            criterion = nn.MSELoss()

            n_batches = len(self.noisy_eeg) // batch_size

            for epoch in range(epochs):
                total_losses = [0.0] * len(self.experts)

                torch.manual_seed(OPTIMAL_SEED + epoch)
                indices = torch.randperm(len(self.noisy_eeg))

                for i in range(n_batches):
                    batch_indices = indices[i * batch_size: (i + 1) * batch_size]

                    batch_noisy = self.noisy_eeg[batch_indices]
                    batch_eog = self.eog[batch_indices]
                    batch_emg = self.emg[batch_indices]
                    batch_clean = self.clean_eeg[batch_indices]

                    ica_features = self.compute_ica_features(batch_noisy, batch_indices=batch_indices)

                    for expert_id, (expert, optimizer) in enumerate(zip(self.experts, optimizers)):
                        optimizer.zero_grad()

                        target = self.create_ica_based_targets(batch_noisy, batch_clean, expert_id)
                        output = expert(batch_noisy, batch_eog, batch_emg, ica_features)

                        loss = criterion(output, target)

                        if expert_id > 0 and self.ica_decomposer.fitted:
                            independence_loss = 0.0
                            for prev_id in range(expert_id):
                                prev_output = self.experts[prev_id](
                                    batch_noisy, batch_eog, batch_emg, ica_features
                                ).detach()

                                correlation = torch.abs(torch.mean(
                                    (output - output.mean()) * (prev_output - prev_output.mean())
                                ) / (output.std() * prev_output.std() + 1e-10))

                                independence_loss += correlation

                            # `lambda_independence` comes from the enclosing
                            # scope of apply_rnn_moe_filter_ica via closure.
                            loss = loss + lambda_independence * independence_loss

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(expert.parameters(), max_norm=1.0)
                        optimizer.step()

                        total_losses[expert_id] += loss.item()

                if epoch % 2 == 0:
                    avg_losses = [loss / n_batches for loss in total_losses]
                    logging.info(f"Pre-training Epoch {epoch}: Expert losses: {avg_losses}")

            logging.info("ICA-based self-learning pre-training completed!")

    class RNNMoEFilter(nn.Module):
        def __init__(self, num_experts, ica_decomposer):
            super().__init__()

            self.experts = nn.ModuleList([RNNExpert(i) for i in range(num_experts)])
            self.num_experts = num_experts
            self.ica_decomposer = ica_decomposer

            input_size = 7
            self.gate_lstm = nn.LSTM(input_size, 64, batch_first=True)
            self.gate_network = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_experts)
            )

            self.sparse_gate = True
            # top_k pulled from enclosing scope of apply_rnn_moe_filter_ica.
            self.top_k = top_k
            # If False, the gating network sees zeros instead of ICA features
            # (ablation A4 in the paper).
            self.use_ica_features_in_gate = use_ica_features_in_gate

        def compute_ica_features(self, signals, batch_indices=None):
            """
            Look up cached ICA features by indices. The cache is built once
            by ICASelfLearningPretrainer and shared via ica_decomposer.features_cache.
            Falls back to a slow per-batch compute if no cache or no indices.
            """
            cache = getattr(self.ica_decomposer, "features_cache", None)
            if batch_indices is not None and cache is not None:
                if torch.is_tensor(batch_indices):
                    batch_indices = batch_indices.detach().cpu().numpy()
                return cache[batch_indices]

            batch_size, seq_len = signals.shape
            n_components = self.ica_decomposer.n_components if self.ica_decomposer.fitted else 4
            ica_features = torch.zeros(batch_size, seq_len, n_components, device=signals.device)
            if not self.ica_decomposer.fitted:
                return ica_features

            window_size = min(64, seq_len)
            stride = 8
            for i in range(batch_size):
                signal_np = signals[i].detach().cpu().numpy()
                weights = np.zeros((seq_len, n_components), dtype=np.float32)
                counts = np.zeros(seq_len, dtype=np.float32)
                for j in range(0, seq_len - window_size + 1, stride):
                    window = signal_np[j:j + window_size] * np.hanning(window_size)
                    components = self.ica_decomposer.decompose(window)
                    if components is None:
                        continue
                    for comp_idx in range(min(n_components, components.shape[0])):
                        comp = components[comp_idx]
                        comp_std = np.std(comp) + 1e-10
                        comp_normalized = comp / comp_std
                        for k in range(j, min(j + window_size, seq_len)):
                            local_idx = int((k - j) * len(comp) / window_size)
                            weight = 1.0 - abs(2.0 * (k - j) / window_size - 1.0)
                            weights[k, comp_idx] += weight * comp_normalized[min(local_idx, len(comp) - 1)]
                            counts[k] += weight
                nz = counts > 0
                if nz.any():
                    block = weights[nz] / counts[nz, None]
                    ica_features[i, np.where(nz)[0], :] = torch.from_numpy(block).to(signals.device)
            ica_features = F.normalize(ica_features, p=2, dim=-1)
            return ica_features

        def forward(self, x, eog, emg, batch_indices=None):
            ica_features = self.compute_ica_features(x, batch_indices=batch_indices)
            basic_features = torch.stack([x, eog, emg], dim=-1)
            # Ablation A4: zero out ICA features at the gating input if requested.
            # The experts still receive ICA features (the ablation isolates the
            # gate's reliance on ICA, not the experts').
            gate_ica = ica_features if self.use_ica_features_in_gate else torch.zeros_like(ica_features)
            gate_input = torch.cat([basic_features, gate_ica], dim=-1)

            gate_lstm_out, _ = self.gate_lstm(gate_input)
            gate_logits = self.gate_network(gate_lstm_out)
            gate_probs = F.softmax(gate_logits, dim=2)

            self.gate_probs = gate_probs

            if self.sparse_gate:
                top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=2)
                mask = torch.zeros_like(gate_probs)
                mask.scatter_(2, top_k_indices, top_k_probs)
                mask = mask / mask.sum(dim=2, keepdim=True).clamp(min=1e-10)
                gate_probs = mask

            expert_outputs = []
            for expert in self.experts:
                expert_out = expert(x, eog, emg, ica_features)
                expert_outputs.append(expert_out)

            stacked_outputs = torch.stack(expert_outputs, dim=2)
            combined_output = torch.sum(stacked_outputs * gate_probs, dim=2)

            return combined_output

        def get_gate_probs(self):
            return self.gate_probs if hasattr(self, 'gate_probs') else None

    def compute_load_balancing_loss(gate_probs):
        """
        Switch-Transformer-style load-balancing auxiliary loss
        (Fedus et al., 2022, Eq. 4 -- 6).

            L_lb = N * sum_e f_e * P_e

        where
            P_e = mean over (batch, time) of gate_probs[..., e]
                  (mean gate probability for expert e)
            f_e = mean over (batch, time) of 1{argmax_k gate_probs[..., k] == e}
                  (fraction of tokens hard-routed to expert e)

        Both P_e and f_e lie in [0, 1] and each sums to 1 over experts, so
        L_lb lies in [1, N]. With N=4 the value at perfect balance is 1.

        BUG-FIX NOTE: the previous implementation used `gate_probs.sum`
        (over batch & time) for f_e, which inflated the loss by
        B * T = 16384 and produced the spurious LB Loss ~ 65536 reported
        during training. The previous numerical scale is preserved by
        re-tuning `lb_coef` upstream if needed (see calling code).
        """
        # P_e: mean gate probability per expert
        P = gate_probs.mean(dim=[0, 1])              # (E,)
        # f_e: fraction of (batch, time) tokens whose argmax is expert e
        hard = gate_probs.argmax(dim=-1)             # (B, T)
        E = P.size(0)
        f = torch.zeros_like(P)
        for e in range(E):
            f[e] = (hard == e).float().mean()
        loss = E * torch.sum(f * P)
        return loss

    # ICA-BASED SELF-LEARNING PRE-TRAINING PHASE
    pretrainer = ICASelfLearningPretrainer(
        [],
        noisy_eeg_tensor,
        eog_tensor,
        emg_tensor,
        clean_tensor_full,
    )

    model = RNNMoEFilter(num_experts, pretrainer.ica_decomposer).to(device)
    pretrainer.experts = model.experts
    if pretrain_epochs > 0:
        pretrainer.pretrain(epochs=pretrain_epochs, batch_size=32)
    else:
        logging.info("Pre-training skipped (pretrain_epochs=0).")

    # Main training phase
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = main_epochs
    batch_size = 32
    n_batches = len(noisy_eeg) // batch_size
    # lb_coef and lambda_independence come from kwargs above (ablation knobs).

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logging.info("Starting main training phase with ICA pre-trained experts...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_lb_loss = 0

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            optimizer.zero_grad()

            batch_noisy = noisy_eeg_tensor[start_idx:end_idx]
            batch_eog = eog_tensor[start_idx:end_idx]
            batch_emg = emg_tensor[start_idx:end_idx]
            batch_target = clean_tensor_full[start_idx:end_idx]
            batch_indices = list(range(start_idx, end_idx))

            output = model(batch_noisy, batch_eog, batch_emg, batch_indices=batch_indices)
            main_loss = criterion(output, batch_target)

            gate_probs = model.get_gate_probs()
            lb_loss = compute_load_balancing_loss(gate_probs) if gate_probs is not None else 0

            loss = main_loss + lb_coef * lb_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += main_loss.item()
            if isinstance(lb_loss, torch.Tensor):
                total_lb_loss += lb_loss.item()

        scheduler.step()

        avg_loss = total_loss / n_batches
        avg_lb_loss = total_lb_loss / n_batches if total_lb_loss != 0 else 0

        if epoch % 2 == 0:
            logging.info(f"Epoch {epoch}, Main Loss: {avg_loss:.6f}, LB Loss: {avg_lb_loss:.6f}")

    # Apply the model -- batch the inference to keep activation memory bounded
    model.eval()
    n_total = noisy_eeg_tensor.shape[0]
    inference_batch = 64
    outputs = []
    with torch.no_grad():
        for s in range(0, n_total, inference_batch):
            e = min(s + inference_batch, n_total)
            indices = list(range(s, e))
            out = model(noisy_eeg_tensor[s:e], eog_tensor[s:e], emg_tensor[s:e],
                        batch_indices=indices)
            outputs.append(out.detach().cpu())
    filtered_output = torch.cat(outputs, dim=0).numpy()
    filtered_output = filtered_output * eeg_stds + eeg_means

    return filtered_output


def apply_rnn_moe_filter_ica_eog_only(noisy_eeg: np.ndarray,
                                       eeg_normalized: np.ndarray,
                                       eog_normalized: np.ndarray,
                                       eeg_stds: np.ndarray,
                                       eeg_means: np.ndarray,
                                       seed: int = 40,
                                       device=None,
                                       **kwargs) -> np.ndarray:
    """
    Apply ICA-MoE filter for EEG+EOG denoising (no EMG).

    Args:
        noisy_eeg: Noisy EEG data (contaminated with EOG only)
        eeg_normalized: Normalized clean EEG
        eog_normalized: Normalized EOG
        eeg_stds: Standard deviations for denormalization
        eeg_means: Means for denormalization
        seed: Random seed

    Returns:
        Filtered EEG data
    """
    # Create zeros for EMG
    emg_normalized = np.zeros_like(eog_normalized)

    return apply_rnn_moe_filter_ica(
        noisy_eeg, eeg_normalized, eog_normalized, emg_normalized,
        eeg_stds, eeg_means, seed, device=device, **kwargs,
    )


def apply_rnn_moe_filter_ica_emg_only(noisy_eeg: np.ndarray,
                                       eeg_normalized: np.ndarray,
                                       emg_normalized: np.ndarray,
                                       eeg_stds: np.ndarray,
                                       eeg_means: np.ndarray,
                                       seed: int = 40,
                                       device=None,
                                       **kwargs) -> np.ndarray:
    """
    Apply ICA-MoE filter for EEG+EMG denoising (no EOG).

    Args:
        noisy_eeg: Noisy EEG data (contaminated with EMG only)
        eeg_normalized: Normalized clean EEG
        emg_normalized: Normalized EMG
        eeg_stds: Standard deviations for denormalization
        eeg_means: Means for denormalization
        seed: Random seed

    Returns:
        Filtered EEG data
    """
    # Create zeros for EOG
    eog_normalized = np.zeros_like(emg_normalized)

    return apply_rnn_moe_filter_ica(
        noisy_eeg, eeg_normalized, eog_normalized, emg_normalized,
        eeg_stds, eeg_means, seed, device=device, **kwargs,
    )
