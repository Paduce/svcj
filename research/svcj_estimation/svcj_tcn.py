# -----------------------------------------------------------------------------
# SVCJ hourly‑returns calibration with an Inception‑CNN encoder + MAF posterior
# -----------------------------------------------------------------------------
# Prerequisites (CPU/GPU):
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # adapt CUDA version
#   pip install nflows numpy pandas tqdm
# -----------------------------------------------------------------------------

from __future__ import annotations
import math, dataclasses, time, warnings
from typing import Tuple, List, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

try:
    from nflows import flows, transforms, distributions
except ImportError as e:
    raise RuntimeError("Please `pip install nflows` before running this script") from e

# -----------------------------------------------------------------------------
# 1 ▸ SVCJ parameters + domain transforms
# -----------------------------------------------------------------------------
@dataclasses.dataclass
class SVCJParams:
    mu: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float
    lam: float
    mu_s: float
    sigma_s: float
    mu_v: float
    sigma_vj: float
    rho_j: float

    # --------------------------------------------------------------
    @staticmethod
    def fields() -> List[str]:
        return list(SVCJParams.__annotations__.keys())

    # --------------------------------------------------------------
    @staticmethod
    def sample_crypto_prior(rng: np.random.Generator | None = None) -> "SVCJParams":
        rng = rng or np.random.default_rng()
        return SVCJParams(
            mu       = rng.uniform(-5e-4, 5e-4),
            kappa    = rng.uniform(0.01, 6.0),
            theta    = rng.uniform(1e-6, 1e-2),
            sigma_v  = rng.uniform(1e-4, 0.1),
            rho      = rng.uniform(-0.4, 0.4),
            lam      = rng.uniform(0.0, 10.0),
            mu_s     = rng.uniform(-0.10, 0.10),
            sigma_s  = rng.uniform(0.01, 0.20),
            mu_v     = rng.uniform(1e-6, 1e-2),
            sigma_vj = rng.uniform(1e-6, 1e-2),
            rho_j    = rng.uniform(-0.4, 0.4),
        )

# --------------------------------------------------------------
# domain ↔ ℝ bijections
# --------------------------------------------------------------
_eps = 1e-6

_DEF_TRANSFORMS: dict[str, Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = {
    # identity
    "mu"      : (lambda z: z, lambda z: z),
    "mu_s"    : (lambda z: z, lambda z: z),
    # positive → ℝ via log1p
    "kappa"   : (np.log1p, lambda t: np.expm1(t)),
    "theta"   : (np.log1p, lambda t: np.expm1(t)),
    "sigma_v" : (np.log1p, lambda t: np.expm1(t)),
    "lam"     : (np.log1p, lambda t: np.expm1(t)),
    "sigma_s" : (np.log1p, lambda t: np.expm1(t)),
    "mu_v"    : (np.log1p, lambda t: np.expm1(t)),
    "sigma_vj": (np.log1p, lambda t: np.expm1(t)),
    # correlations (‑1,1) → ℝ via scaled logit
    "rho"     : (
        lambda z: np.log(np.clip((z + 1)/2, _eps, 1-_eps)) - np.log1p(-np.clip((z + 1)/2, _eps, 1-_eps)),
        lambda t: 2*(1/(1+np.exp(-t)))-1,
    ),
    "rho_j"   : (
        lambda z: np.log(np.clip((z + 1)/2, _eps, 1-_eps)) - np.log1p(-np.clip((z + 1)/2, _eps, 1-_eps)),
        lambda t: 2*(1/(1+np.exp(-t)))-1,
    ),
}

_PARAMS_ORDER = SVCJParams.fields()


def params_to_array(p: SVCJParams) -> np.ndarray:
    return np.asarray(dataclasses.astuple(p), dtype=np.float32)


def transform_params(p: np.ndarray) -> np.ndarray:
    out = []
    for val, name in zip(p, _PARAMS_ORDER):
        fwd, _ = _DEF_TRANSFORMS[name]
        out.append(fwd(val))
    return np.asarray(out, dtype=np.float32)


def inv_transform_params(t: torch.Tensor | np.ndarray) -> torch.Tensor:
    arr = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t
    out: list[np.ndarray] = []
    for val, name in zip(arr, _PARAMS_ORDER):
        _, inv = _DEF_TRANSFORMS[name]
        out.append(inv(val))
    return torch.from_numpy(np.asarray(out, dtype=np.float32))

# -----------------------------------------------------------------------------
# 2 ▸ SVCJ path simulator (vectorised Euler–Maruyama)
# -----------------------------------------------------------------------------
class SVCJSimulator:
    @staticmethod
    def simulate(params: SVCJParams, steps: int, *, dt: float = 1/24, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        mu, kappa, theta, sigma_v, rho, lam, mu_s, sig_s, mu_v, sig_vj, _ = dataclasses.astuple(params)
        # draw all randomness at once (⇡ speed)
        dW_s = rng.normal(0.0, math.sqrt(dt), size=steps)
        dW_v = rho * dW_s + math.sqrt(max(1e-8, 1 - rho ** 2)) * rng.normal(0.0, math.sqrt(dt), size=steps)
        dN   = rng.poisson(lam * dt, size=steps)
        Z_s  = rng.normal(mu_s,  sig_s , size=steps) * dN
        Z_v  = rng.normal(mu_v,  sig_vj, size=steps) * dN
        V = theta
        rets = np.empty(steps, dtype=np.float32)
        for t in range(steps):
            V_pos = max(V, 0.0)
            rets[t] = mu*dt - 0.5*V_pos*dt + math.sqrt(V_pos)*dW_s[t] + Z_s[t]
            V = max(V + kappa*(theta - V)*dt + sigma_v*math.sqrt(V_pos)*dW_v[t] + Z_v[t], 1e-12)
        return rets

# -----------------------------------------------------------------------------
# 3 ▸ Dataset
# -----------------------------------------------------------------------------
class SVCJDataset(Dataset):
    def __init__(self, n_samples: int, *, window: int = 24*14, rng: np.random.Generator | None = None):
        self.n = n_samples
        self.T = window
        self.rng = rng or np.random.default_rng()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        psi  = SVCJParams.sample_crypto_prior(self.rng)
        rets = SVCJSimulator.simulate(psi, self.T, rng=self.rng)
        x    = torch.from_numpy(rets).float().unsqueeze(0)            # [1,T]
        theta_true = transform_params(params_to_array(psi))           # → ℝ^11
        y    = torch.from_numpy(theta_true)
        return x, y

# -----------------------------------------------------------------------------
# 4 ▸ Inception‑style encoder (multi‑scale CNN)
# -----------------------------------------------------------------------------
class InceptionBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        branch_c = out_c // 4
        self.branches = nn.ModuleList([
            nn.Conv1d(in_c, branch_c, 3, padding=1),
            nn.Conv1d(in_c, branch_c, 5, padding=2),
            nn.Conv1d(in_c, branch_c, 11, padding=5),
            nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_c, branch_c, 1)),
        ])
        self.bottleneck = nn.Conv1d(out_c, out_c, 1)
        self.relu = nn.ReLU()
        self.res  = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        y = torch.cat([b(x) for b in self.branches], dim=1)
        y = self.bottleneck(y)
        return self.relu(y + self.res(x))

class InceptionEncoder(nn.Module):
    def __init__(self, in_c: int = 1, hidden_c: int = 64, n_blocks: int = 6):
        super().__init__()
        layers: List[nn.Module] = []
        c = in_c
        for _ in range(n_blocks):
            layers.append(InceptionBlock(c, hidden_c))
            c = hidden_c
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):                 # x: [B,1,T]
        h = self.net(x)
        return self.pool(h).squeeze(-1)    # [B, hidden_c]

# -----------------------------------------------------------------------------
# 5 ▸ Conditional MAF posterior
# -----------------------------------------------------------------------------
class ParamPosterior(nn.Module):
    def __init__(self, code_dim: int, theta_dim: int, n_layers: int = 5, hidden: int = 128):
        super().__init__()
        base = distributions.StandardNormal([theta_dim])
        t_list = []
        for _ in range(n_layers):
            t_list.append(
                transforms.MaskedAffineAutoregressiveTransform(
                    features=theta_dim,
                    hidden_features=hidden,
                    context_features=code_dim,
                )
            )
            t_list.append(transforms.RandomPermutation(features=theta_dim))
        self.flow = flows.Flow(transform=transforms.CompositeTransform(t_list), distribution=base)

    def log_prob(self, theta: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.flow.log_prob(inputs=theta, context=context)

    def sample(self, context: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        return self.flow.sample(num_samples, context=context)         # [num_samples, ⋯]

# -----------------------------------------------------------------------------
# 6 ▸ Full calibration network
# -----------------------------------------------------------------------------
class CalibNet(nn.Module):
    def __init__(self, code_dim: int = 64, theta_dim: int = 11):
        super().__init__()
        self.encoder = InceptionEncoder(hidden_c=code_dim)
        self.posterior = ParamPosterior(code_dim, theta_dim)

    # ------------------------------------------------------------------
    def log_prob(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)                    # [B, code_dim]
        return self.posterior.log_prob(theta, context=h)

    # ------------------------------------------------------------------
    def posterior_sample(self, x: torch.Tensor, n: int = 1) -> torch.Tensor:
        h = self.encoder(x)
        return self.posterior.sample(context=h, num_samples=n)        # [n,B,theta_dim]

# -----------------------------------------------------------------------------
# 7 ▸ Training helpers
# -----------------------------------------------------------------------------
class Trainer:
    def __init__(self, model: CalibNet, *, lr: float = 3e-4, device: str | torch.device = "cuda") -> None:
        self.model = model.to(device)
        self.opt   = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        self.dev   = device

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        tot, n = 0.0, 0
        for x, theta in loader:
            x, theta = x.to(self.dev, non_blocking=True), theta.to(self.dev, non_blocking=True)
            loss = -self.model.log_prob(x, theta).mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            tot += loss.item()*x.size(0); n += x.size(0)
        return tot / n

# -----------------------------------------------------------------------------
# 8 ▸ Inference utility
# -----------------------------------------------------------------------------
class SVCJInference:
    def __init__(self, model: CalibNet, device: str | torch.device = "cpu"):
        self.model = model.eval().to(device)
        self.dev = device

    @torch.no_grad()
    def estimate(self, returns: np.ndarray | torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(returns, np.ndarray):
            x = torch.from_numpy(returns).float()
        else:
            x = returns.float()
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # [1,1,T]
        elif x.ndim == 2:
            x = x.unsqueeze(1)
        x = x.to(self.dev)
        samples = self.model.posterior_sample(x, n=n_samples)  # [n,B,11]
        mean = samples.mean(0).squeeze(0)                     # [11]
        cov  = torch.cov(samples.squeeze(1).T)                # [11,11]
        return inv_transform_params(mean), cov

# -----------------------------------------------------------------------------
# 9 ▸ Main training script (example)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    DEVICE = "cuda"
    T_WINDOW = 24*14          # 14 days of hourly returns
    BATCH    = 1024
    STEPS    = 400            # synthetic samples per epoch = STEPS*BATCH
    EPOCHS   = 100

    ds = SVCJDataset(n_samples=STEPS*BATCH, window=T_WINDOW)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)

    net = CalibNet(code_dim=64, theta_dim=11)
    trainer = Trainer(net, device=DEVICE)

    for ep in tqdm(range(1, EPOCHS+1)):
        t0 = time.time()
        loss = trainer.train_epoch(dl)
        print(f"[epoch {ep:03d}] loss={loss:6.4f}  ({time.time()-t0:4.1f}s)")

    torch.save(net.state_dict(), "svcj_flow_calib.pt")
    print("Model saved → svcj_flow_calib.pt")

    # quick sanity check on one fresh path
    test_len = 24*10
    test_params = SVCJParams.sample_crypto_prior()
    test_rets = SVCJSimulator.simulate(test_params, test_len)
    infer = SVCJInference(net, DEVICE)
    est_mean, est_cov = infer.estimate(test_rets, n_samples=500)

    print("\nTrue parameters vs posterior mean (first 5 shown):")
    for name, tval, pval in zip(_PARAMS_ORDER[:5], params_to_array(test_params)[:5], est_mean[:5]):
        print(f"  {name:8s}: true={tval:+.4e}   est={pval.item():+.4e}")

    warnings.warn("This is a minimal demo. Tune hyper‑params & add validation for production.")
