from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from utils.dataset import DICHASUSDataset, DeepmimoO1Dataset


class LMMSEModel:
    """
    LMMSE channel estimator with optional sample/global noise scaling.

    Inference flow:
    1) Vectorize observed partial channel `y`.
    2) Build noise covariance `Sigma_n` from SNR (or fallback regularization).
    3) Solve `(R_yy + Sigma_n) z = y` and recover `h_est = R_xy z`.
    4) Reshape estimate back to `[B, Ant, Car, 2]`.

    Formula used:
        x_hat = R_xy @ (R_yy + Sigma_n)^(-1) @ y
    where:
        R_xy = E[x y^H],  R_yy = E[y y^H] (signal-only part).
    """

    def __init__(self, R_xx, input_indices, num_ant, num_car, noise_mode='sample'):
        """
        Args:
            R_xx: Global channel covariance [N_ant*N_car, N_ant*N_car].
            input_indices: Observed-position indices in flattened vector [N_input].
            num_ant: Number of antennas.
            num_car: Number of carriers.
            noise_mode: 'sample' or 'global'.
        """
        self.device = R_xx.device
        self.num_ant = num_ant
        self.num_car = num_car
        self.noise_mode = noise_mode

        # Cached covariance terms used in every inference call.
        # With observation y = Phi x + n:
        #   R_xy = E[x y^H] = R_xx Phi^H
        #   R_yy(signal) = Phi R_xx Phi^H
        # Here Phi is implemented implicitly via index selection.
        self.R_xx = R_xx.cfloat()
        self.input_indices = input_indices.long()
        self.R_xy = self.R_xx[:, self.input_indices]          # [M, K]
        self.R_yy_signal = self.R_xy[self.input_indices, :]   # [K, K]

        # Optional global antenna power baseline for noise estimation.
        # diag(R_xx) approximates E[|x|^2] per (ant, car) entry.
        self.global_ant_power = None
        if self.noise_mode == 'global':
            diag_R = torch.diagonal(self.R_xx).real
            ant_power = diag_R.view(num_ant, num_car).mean(dim=1)  # [Ant]
            self.global_ant_power = ant_power.to(self.device)
            print(f"LMMSE Noise Mode: 'global' (Pre-calculated antenna power: {self.global_ant_power.shape})")
        else:
            print("LMMSE Noise Mode: 'sample' (Adaptive to input instance power)")

    def to(self, device):
        """Move cached tensors to a target device (nn.Module-like API)."""
        self.device = device
        self.R_xx = self.R_xx.to(device)
        self.input_indices = self.input_indices.to(device)
        self.R_xy = self.R_xy.to(device)
        self.R_yy_signal = self.R_yy_signal.to(device)
        if self.global_ant_power is not None:
            self.global_ant_power = self.global_ant_power.to(device)
        return self

    def eval(self):
        """Compatibility stub for nn.Module.eval()."""
        pass

    def __call__(self, h_now_input, h_past_input=None, snr_now=None):
        """
        Args:
            h_now_input: Observed partial channel [B, Ant_in, Car_in, 2].
            h_past_input: Unused; kept for API compatibility.
            snr_now: Per-antenna SNR in dB [B, Ant].
        Returns:
            Estimated channel [B, Ant, Car, 2].
        """
        if h_now_input.device != self.device:
            h_now_input = h_now_input.to(self.device)

        B = h_now_input.shape[0]
        K = self.input_indices.shape[0]

        # Step 1: vectorize observed channel to complex vector y.
        y = torch.view_as_complex(h_now_input).reshape(B, -1)  # [B, K]

        # Step 2: build per-sample noise covariance Sigma_n.
        # Assumption: noise is spatially uncorrelated across observed entries,
        # so Sigma_n is diagonal.
        if snr_now is not None:
            # Align SNR to the observed antenna subset.
            ant_indices_all = self.input_indices // self.num_car
            ant_indices_unique = torch.unique(ant_indices_all, sorted=True)
            if snr_now.device != self.device:
                snr_now = snr_now.to(self.device)
            snr_input = snr_now[:, ant_indices_unique]  # [B, Ant_in]
            snr_linear = 10 ** (snr_input / 10.0)

            # Convert SNR to noise power via:
            #   SNR = P_signal / P_noise,  P_total = P_signal + P_noise
            # => P_noise = P_total / (1 + SNR).
            if self.noise_mode == 'sample':
                # Sample mode: derive P_total from current sample energy.
                # + Adapts to path-loss changes.
                # - Can underestimate noise in deep fades.
                # Shape: [B, 1] (later broadcast over observed antennas).
                power_basis = torch.mean(torch.abs(y) ** 2, dim=1, keepdim=True)

            elif self.noise_mode == 'global':
                # Global mode: use dataset-level average antenna power.
                # + Stable noise floor; robust to deep fades.
                # - Less adaptive when path-loss spread is large.
                # - Works best when SNR estimates are reliable.

                # Select observed antennas: [Ant_total] -> [Ant_in].
                global_power_selected = self.global_ant_power[ant_indices_unique]
                # Broadcast to batch: [1, Ant_in] -> [B, Ant_in].
                power_basis = global_power_selected.unsqueeze(0).expand(B, -1)

            else:
                raise ValueError(f"Unknown noise_mode: {self.noise_mode}")

            # power_basis is interpreted as total (signal + noise) power.
            # This is consistent with global mode because R_xx is also
            # estimated from noisy observations.
            noise_power_abs = power_basis / (1.0 + snr_linear)  # [B, Ant_in]
            num_car_in = h_now_input.shape[2]
            # Broadcast per-antenna noise power across observed carriers.
            noise_var_flat = noise_power_abs.unsqueeze(-1).repeat(1, 1, num_car_in).reshape(B, -1)
            Sigma_n = torch.diag_embed(noise_var_flat).cfloat()  # [B, K, K]
        else:
            # If SNR is unavailable, use a small isotropic regularizer.
            Sigma_n = torch.eye(K, device=self.device).unsqueeze(0).repeat(B, 1, 1).cfloat() * 1e-4

        # Step 3: LMMSE solve.
        #   h_est = R_xy (R_yy + Sigma_n)^-1 y
        # Using `torch.linalg.solve` is numerically safer than explicit matrix inverse.
        R_total = self.R_yy_signal.unsqueeze(0) + Sigma_n
        z = torch.linalg.solve(R_total, y.unsqueeze(-1))       # [B, K, 1]
        h_est_flat = torch.matmul(self.R_xy, z).squeeze(-1)    # [B, M]

        # Step 4: restore [B, Ant, Car, 2].
        h_est = h_est_flat.view(B, self.num_ant, self.num_car)
        h_est = torch.view_as_real(h_est)

        return h_est


def get_lmmse_model(conf):
    """
    Build an `LMMSEModel` by estimating covariance from dataset samples.

    Initialization flow:
    1) Create a temporary dataset loader.
    2) Estimate global covariance R_xx from flattened complex channels.
    3) Build observed-index mask from input antenna/carrier sampling pattern.
    4) Instantiate and return `LMMSEModel`.

    Covariance estimator:
        R_xx ≈ (1/N) Σ x_i x_i^H
    """
    print("====== Initializing LMMSE Model ======")

    lmmse_conf = conf.get('lmmse', {})
    noise_mode = lmmse_conf.get('noise_mode', 'sample')
    print(f"Configured Noise Estimation Mode: {noise_mode}")

    # Use available accelerator for one-time covariance estimation.
    calc_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Calculating Correlation Matrix (R_xx) on {calc_device}...")

    dichasus_conf = conf.get('dichasus', {})
    train_h5 = dichasus_conf.get('train_dataset_name', conf.get('train_dataset_name'))
    test_h5 = dichasus_conf.get('test_dataset_name', conf.get('test_dataset_name'))
    data_dir = conf['data_dirs'][0]

    def _create_temp_loader(d_dir, d_name):
        dataset_type = conf['dataset_type']
        if dataset_type == 'DICHASUS':
            ds = DICHASUSDataset(d_dir, dataset_name=d_name, max_samples=None, load_to_memory=True)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        if len(ds) == 0:
            return None
        return DataLoader(ds, batch_size=100, shuffle=False)

    if train_h5:
        loader = _create_temp_loader(data_dir, d_name=train_h5)
    else:
        print("Warning: No training set specified. Using Test set for Covariance.")
        loader = _create_temp_loader(data_dir, d_name=test_h5)

    # Estimate R_xx = E[x x^H] from all available samples.
    # This empirical covariance is the key prior used by LMMSE.
    num_ant = conf['num_ant']
    num_car = conf['num_car']
    total_features = num_ant * num_car

    R_acc = torch.zeros((total_features, total_features), device=calc_device, dtype=torch.complex64)
    N_samples = 0

    for batch in tqdm(loader, desc="Calc Covariance"):
        csi_data = batch if isinstance(batch, torch.Tensor) else batch['csi']
        B, S, Ant, Car, C = csi_data.shape
        csi_data = csi_data.reshape(B * S, Ant, Car, C)

        csi = torch.view_as_complex(csi_data.to(calc_device).float())
        B = csi.shape[0]
        flat = csi.view(B, -1)

        # Accumulate X^H X for covariance estimation.
        R_batch = torch.einsum('bi,bj->ij', flat, flat.conj())
        R_acc += R_batch
        N_samples += B

    R_xx = R_acc / N_samples
    print(f"Computed R_xx with {N_samples} samples. (Absolute Power Scale)")

    # Build observed position indices from sparse input grid.
    step_ant = conf['num_ant'] // conf['input_ant_size']
    step_car = conf['num_car'] // conf['input_car_size']

    indices_list = []
    idx_ant = list(range(0, step_ant * (conf['input_ant_size'] - 1) + 1, step_ant))
    idx_car = list(range(0, step_car * (conf['input_car_size'] - 1) + 1, step_car))

    for a in idx_ant:
        for c in idx_car:
            flat_idx = a * num_car + c
            indices_list.append(flat_idx)

    input_indices = torch.tensor(indices_list, dtype=torch.long, device=calc_device)
    print(f"Input Indices count: {len(input_indices)}")

    # build and return estimator (currently on calc_device).
    model = LMMSEModel(R_xx, input_indices, num_ant, num_car, noise_mode=noise_mode)
    return model



