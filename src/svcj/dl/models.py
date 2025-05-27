"""
Enhanced neural network models for SV family parameter estimation.
Coherent with the improved training and dataset modules.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Literal, Optional, Dict, Any
import warnings

__all__ = [
    "CNNRegressor", "CNN_Ficura", "TCNRegressor", "TransformerRegressor",
    "MLPRegressor",  # Added MLPRegressor
    "build_model", "ModelConfig", "validate_model_output"
]


class ModelConfig:
    """Configuration for SV family models."""

    def __init__(
        self,
        n_params: int,
        model_type: str = "svcj",
        arch_type: str = "cnn",
        use_heteroscedastic_loss: bool = True,
        dropout_rate: float = 0.1,
        hidden_dim: int = 64,
        n_layers: int = 3,
        # Added for MLP
        mlp_hidden_dims: Optional[list[int]] = None,
        input_dim: Optional[int] = None, # For MLP, this will be seq_len
    ):
        self.n_params = n_params
        self.model_type = model_type.lower()
        self.arch_type = arch_type.lower()
        self.use_heteroscedastic_loss = use_heteroscedastic_loss
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim # General hidden_dim, MLP uses mlp_hidden_dims
        self.n_layers = n_layers # General n_layers, MLP uses len(mlp_hidden_dims)

        # MLP specific
        self.mlp_hidden_dims = mlp_hidden_dims if mlp_hidden_dims is not None else [128, 64]
        self.input_dim = input_dim

        self.validate()

    def validate(self):
        """Validate model configuration."""
        if self.n_params <= 0:
            raise ValueError(f"n_params must be positive, got {self.n_params}")
        if self.model_type not in ("sv", "svj", "svcj"):
            raise ValueError(f"model_type must be one of ['sv', 'svj', 'svcj'], got {self.model_type}")
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError(f"dropout_rate must be in [0, 1], got {self.dropout_rate}")


def _conv_block(
    in_ch: int,
    out_ch: int,
    k: int = 5,
    pool: bool = True,
    pool_stride: int = 5,
    first_pool: bool = False,
    negative_slope: float = 1e-2,
    dropout_rate: float = 0.0
) -> list[nn.Module]:
    """Create a convolutional block with optional pooling and dropout."""

    # Use proper padding to maintain sequence length
    padding = k // 2

    layers: list[nn.Module] = [
        nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=padding),
        nn.LeakyReLU(negative_slope=negative_slope),
        nn.GroupNorm(1, out_ch),
    ]

    if dropout_rate > 0:
        layers.append(nn.Dropout1d(dropout_rate))

    if pool:
        stride = 1 if first_pool else pool_stride
        # Ensure pooling doesn't reduce sequence too much
        pool_size = min(5, k)
        pool_padding = pool_size // 2
        layers.append(nn.AvgPool1d(kernel_size=pool_size, stride=stride, padding=pool_padding))

    return layers


class BaseSVModel(nn.Module):
    """Base class for SV family models with common functionality."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_params = config.n_params

        # Will be set by subclasses
        self.backbone = None
        self.feature_dim = None

    def _handle_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standardize input handling across all models.
        Expected input: (B, 1, L) from dataset for CNN/TCN/Transformer
        For MLP, expected input: (B, L) from dataset, will be flattened.
        Returns: (B, 1, L) for conv layers or (B, L) for MLP.
        """
        if self.config.arch_type == "mlp":
            if x.dim() == 3 and x.shape[1] == 1:
                # (B, 1, L) -> (B, L)
                x = x.squeeze(1)
            elif x.dim() != 2:
                 raise ValueError(f"Expected 2D or 3D (B,1,L) input for MLP, got {x.dim()}D")
        else: # CNN, TCN, Transformer
            if x.dim() == 2:
                # (B, L) -> (B, 1, L)
                x = x.unsqueeze(1)
            elif x.dim() == 3:
                if x.shape[1] != 1:
                    raise ValueError(f"Expected channel dimension of 1, got {x.shape[1]}")
            else:
                raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        return x

    def _compute_mu_features(self, x: torch.Tensor, mu_mode: str) -> torch.Tensor:
        """
        Compute features for mu prediction based on mode.

        Args:
            x: Input tensor of shape (B, L) or (B, 1, L)
            mu_mode: One of "raw", "analytical", "devol"

        Returns:
            Features tensor for mu prediction
        """
        # Ensure x_flat is (B, L)
        if x.dim() == 3 and x.shape[1] == 1:
            x_flat = x.squeeze(1)  # (B, L)
        elif x.dim() == 2:
            x_flat = x # (B,L)
        else:
            raise ValueError(f"Input to _compute_mu_features has unexpected shape: {x.shape}")


        sample_mean = x_flat.mean(dim=1, keepdim=True)  # (B, 1)
        sample_std = x_flat.std(dim=1, unbiased=False, keepdim=True) + 1e-8  # (B, 1)

        if mu_mode == "devol":
            # Use both raw mean and standardized mean
            z_score_mean = sample_mean / sample_std
            return torch.cat([sample_mean, z_score_mean], dim=1)  # (B, 2)
        elif mu_mode == "raw":
            # Use raw mean + dummy zero feature
            zero = torch.zeros_like(sample_mean)
            return torch.cat([sample_mean, zero], dim=1)  # (B, 2)
        elif mu_mode == "analytical":
            # Return the empirical mean directly (will be used as mu prediction)
            return sample_mean  # (B, 1)
        else:
            raise ValueError(f"Unknown mu_mode: {mu_mode}")

    def _create_heads(self, feature_dim: int, n_heads: int) -> nn.ModuleList:
        """Create parameter prediction heads."""

        return nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, self.config.hidden_dim),
                nn.LeakyReLU(1e-2),
                nn.LayerNorm(self.config.hidden_dim),
                nn.Dropout(self.config.dropout_rate),
                nn.Linear(self.config.hidden_dim, 1)
            )
            for _ in range(n_heads)
        ])

    def forward(self, x: torch.Tensor, mu_mode: str = "raw") -> torch.Tensor:
        """Forward pass - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward method")

    def get_effective_params(self, mu_mode: str) -> int:
        """Get the effective number of parameters for given mu_mode."""
        if mu_mode == "analytical":
            return self.n_params - 1  # Exclude mu
        else:
            return self.n_params


class CNNRegressor(BaseSVModel):
    """
    Enhanced CNN regressor with proper input handling and mu_mode support.

    Architecture:
    - 3 convolutional blocks with GroupNorm and optional dropout
    - Global average pooling
    - Separate heads for each parameter with proper mu_mode handling
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Backbone: 3 conv blocks → global avg pool
        convs: list[nn.Module] = []
        convs += _conv_block(1, 32, k=7, first_pool=True, dropout_rate=config.dropout_rate)
        convs += _conv_block(32, 64, k=5, dropout_rate=config.dropout_rate)
        convs += _conv_block(64, 128, k=3, pool=False, dropout_rate=config.dropout_rate)
        convs += [nn.AdaptiveAvgPool1d(1), nn.Flatten()]

        self.backbone = nn.Sequential(*convs)
        self.feature_dim = 128

        # Specialized mu head (takes backbone features + mu-specific features)
        mu_feature_dim = 2  # from _compute_mu_features
        self.mu_head = nn.Sequential(
            nn.Linear(self.feature_dim + mu_feature_dim, self.config.hidden_dim),
            nn.LeakyReLU(1e-2),
            nn.LayerNorm(self.config.hidden_dim),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_dim, 1)
        )

        # Heads for remaining parameters
        self.param_heads = self._create_heads(self.feature_dim, self.n_params - 1)

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """Initialize weights using Kaiming normal for better convergence."""
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mu_mode: str = "raw") -> torch.Tensor:
        """
        Forward pass with proper mu_mode handling.

        Args:
            x: Input tensor of shape (B, 1, L)
            mu_mode: Parameter prediction mode

        Returns:
            Parameter predictions of shape (B, n_effective_params)
        """
        # Standardize input
        original_input_shape = x.shape
        x = self._handle_input(x)  # (B, 1, L)

        # Extract backbone features
        backbone_features = self.backbone(x)  # (B, feature_dim)

        # Handle mu prediction based on mode
        if mu_mode == "analytical":
            # Use empirical mean as mu, don't predict it
            mu_features = self._compute_mu_features(x, mu_mode)  # (B, 1)
            mu_pred = mu_features  # (B, 1)

            # Predict only the remaining parameters
            other_preds = torch.cat([head(backbone_features) for head in self.param_heads], dim=1)

            # Return only non-mu parameters for analytical mode
            return other_preds  # (B, n_params-1)
        else:
            # Predict mu using specialized head
            mu_features = self._compute_mu_features(x, mu_mode)  # (B, 2)
            mu_input = torch.cat([backbone_features, mu_features], dim=1)
            mu_pred = self.mu_head(mu_input)  # (B, 1)

            # Predict remaining parameters
            other_preds = torch.cat([head(backbone_features) for head in self.param_heads], dim=1)

            # Return all parameters
            return torch.cat([mu_pred, other_preds], dim=1)  # (B, n_params)


class CNN_Ficura(BaseSVModel):
    """
    Ficura-style CNN with enhanced robustness and proper dimension handling.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Backbone (similar to original but enhanced)
        convs: list[nn.Module] = []
        convs += _conv_block(1, 20, k=5, first_pool=True, dropout_rate=config.dropout_rate)
        convs += _conv_block(20, 40, k=5, dropout_rate=config.dropout_rate)
        convs += _conv_block(40, 60, k=3, pool=False, dropout_rate=config.dropout_rate)
        convs += [nn.AdaptiveAvgPool1d(1), nn.Flatten()]

        self.backbone = nn.Sequential(*convs)
        self.feature_dim = 60

        # Mu head with enhanced features
        mu_feature_dim = 2
        self.mu_head = nn.Sequential(
            nn.Linear(self.feature_dim + mu_feature_dim, 16),
            nn.LeakyReLU(1e-2),
            nn.LayerNorm(16),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(16, 1)
        )

        # Parameter heads
        self.param_heads = self._create_heads(self.feature_dim, self.n_params - 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mu_mode: str = "raw") -> torch.Tensor:
        # Standardize input handling
        original_input_shape = x.shape
        x = self._handle_input(x)

        # Backbone feature extraction
        features = self.backbone(x)  # (B, feature_dim)

        if mu_mode == "analytical":
            # Use empirical mean, predict only remaining parameters
            mu_features = self._compute_mu_features(x, mu_mode)  # (B, 1)
            other_preds = torch.cat([head(features) for head in self.param_heads], dim=1)
            return other_preds  # (B, n_params-1)
        else:
            # Predict mu using network
            mu_features = self._compute_mu_features(x, mu_mode)  # (B, 2)
            mu_input = torch.cat([features, mu_features], dim=1)
            mu_pred = self.mu_head(mu_input)  # (B, 1)

            # Predict remaining parameters
            other_preds = torch.cat([head(features) for head in self.param_heads], dim=1)

            return torch.cat([mu_pred, other_preds], dim=1)  # (B, n_params)


class Chomp1d(nn.Module):
    """Remove padding from temporal convolution to maintain causality."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


def _tcn_block(inp: int, out: int, k: int, d: int, dropout_rate: float = 0.0) -> nn.Module:
    """Create a TCN residual block with proper normalization and dropout."""

    layers = [
        nn.Conv1d(inp, out, k, padding=(k-1)*d, dilation=d),
        Chomp1d((k-1)*d),
        nn.LeakyReLU(1e-2),
        nn.GroupNorm(1, out),
    ]

    if dropout_rate > 0:
        layers.append(nn.Dropout1d(dropout_rate))

    layers.extend([
        nn.Conv1d(out, out, k, padding=(k-1)*d, dilation=d),
        Chomp1d((k-1)*d),
        nn.LeakyReLU(1e-2),
        nn.GroupNorm(1, out),
    ])

    if dropout_rate > 0:
        layers.append(nn.Dropout1d(dropout_rate))

    return nn.Sequential(*layers)


class TCNRegressor(BaseSVModel):
    """
    Enhanced Temporal Convolutional Network for SV parameter estimation.
    """

    def __init__(self, config: ModelConfig, n_levels: int = 4, k: int = 3):
        super().__init__(config)

        self.n_levels = n_levels
        self.k = k

        # TCN layers with exponentially increasing dilation
        tcn_layers = []
        residual_layers = []

        in_ch = 1
        base_ch = 32

        for i in range(n_levels):
            out_ch = base_ch * (2 ** min(i, 2))  # Cap channel growth more aggressively
            dilation = 2 ** i

            tcn_block_module = _tcn_block(in_ch, out_ch, k, dilation, config.dropout_rate)
            tcn_layers.append(tcn_block_module)

            # Residual connection if input/output channels match
            if in_ch != out_ch:
                residual_layers.append(nn.Conv1d(in_ch, out_ch, 1))
            else:
                residual_layers.append(nn.Identity())

            in_ch = out_ch

        self.tcn_layers = nn.ModuleList(tcn_layers)
        self.residual_layers = nn.ModuleList(residual_layers)
        self.feature_dim = in_ch

        # Final pooling and projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, self.config.hidden_dim),
            nn.LeakyReLU(1e-2),
            nn.Dropout(self.config.dropout_rate)
        )

        # Mu head
        mu_feature_dim = 2
        self.mu_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim + mu_feature_dim, self.config.hidden_dim // 2),
            nn.LeakyReLU(1e-2),
            nn.LayerNorm(self.config.hidden_dim // 2),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_dim // 2, 1)
        )

        # Parameter heads
        self.param_heads = self._create_heads(self.config.hidden_dim, self.n_params - 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mu_mode: str = "raw") -> torch.Tensor:
        # Store original input for mu feature computation
        original_input = self._handle_input(x)  # (B, 1, L)

        # TCN forward pass with residual connections
        current = original_input
        for tcn_layer, residual_layer in zip(self.tcn_layers, self.residual_layers):
            residual = residual_layer(current)
            tcn_out = tcn_layer(current)
            current = tcn_out + residual

        # Global pooling and feature projection
        pooled = self.global_pool(current)  # (B, feature_dim, 1)
        features = self.feature_proj(pooled)  # (B, hidden_dim)

        if mu_mode == "analytical":
            # Use empirical mean from original input
            mu_features = self._compute_mu_features(original_input, mu_mode)  # (B, 1)
            other_preds = torch.cat([head(features) for head in self.param_heads], dim=1)
            return other_preds  # (B, n_params-1)
        else:
            # Predict mu using original input features
            mu_features = self._compute_mu_features(original_input, mu_mode)  # (B, 2)
            mu_input = torch.cat([features, mu_features], dim=1)
            mu_pred = self.mu_head(mu_input)  # (B, 1)

            # Predict remaining parameters
            other_preds = torch.cat([head(features) for head in self.param_heads], dim=1)

            return torch.cat([mu_pred, other_preds], dim=1)  # (B, n_params)




class TransformerRegressor(BaseSVModel):
    """
    Transformer-based regressor for SVJD parameter estimation.
    Uses patch embedding, learned [CLS] token, and multi-head attention.

    Args:
        config: ModelConfig with attributes:
            - input_dim (int): original sequence length T
            - hidden_dim (int): d_model
            - dropout_rate (float)
            - n_params (int): total number of model parameters to predict
        n_heads (int): number of attention heads
        n_layers (int): number of transformer blocks
        patch_size (int): kernel size for strided Conv1d
        patch_stride (int): stride for strided Conv1d
    """
    def __init__(
        self,
        config,
        n_heads: int = 8,
        n_layers: int = 4,
        patch_size: int = 8,
        patch_stride: int = 4
    ):
        super().__init__(config)

        self.d_model = config.hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        T = config.input_dim

        # Patch embedding: reduce length by stride
        self.patch_embed = nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model,
            kernel_size=patch_size,
            stride=patch_stride,
            padding=0
        )

        # Learned [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.1)

        # Positional encoding for sequence length after patching + cls
        max_patches = (T - patch_size) // patch_stride + 1
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_patches + 1, self.d_model) * 0.1
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=config.dropout_rate,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # LayerNorm on final token
        self.norm = nn.LayerNorm(self.d_model)

        # μ-head: takes cls output + optional analytical features
        mu_feature_dim = 2
        self.mu_head = nn.Sequential(
            nn.Linear(self.d_model + mu_feature_dim, self.d_model // 2),
            nn.LeakyReLU(1e-2),
            nn.LayerNorm(self.d_model // 2),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.d_model // 2, 1)
        )

        # Other parameter heads
        self.param_heads = self._create_heads(self.d_model, self.n_params - 1)

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mu_mode: str = "raw") -> torch.Tensor:
        # original input for analytical features
        original_input = self._handle_input(x)  # (B,1,T)

        B, C, T = original_input.shape

        # Patch embedding
        x_patch = self.patch_embed(original_input)   # (B, d_model, L')
        Lp = x_patch.size(-1)

        # Sequence for transformer: (B, L'+1, d_model)
        x_seq = x_patch.transpose(1, 2)             # (B, L', d_model)
        cls = self.cls_token.expand(B, -1, -1)      # (B, 1, d_model)
        x_in = torch.cat([cls, x_seq], dim=1)       # (B, L'+1, d_model)

        # Add positional encoding
        if x_in.size(1) <= self.pos_encoding.size(1):
            x_in = x_in + self.pos_encoding[:, : x_in.size(1), :]
        else:
            # (unlikely) interpolate
            pos = F.interpolate(
                self.pos_encoding.transpose(1, 2), 
                size=x_in.size(1), mode='linear', align_corners=False
            ).transpose(1, 2)
            x_in = x_in + pos

        # Transformer encoding
        H = self.transformer(x_in)  # (B, L'+1, d_model)
        cls_out = self.norm(H[:, 0, :])  # (B, d_model)

        # μ-features
        mu_feats = self._compute_mu_features(original_input, mu_mode)  # (B,1) or (B,2)

        # μ prediction
        if mu_mode == "analytical":
            # skip mu head
            other = torch.cat([h(cls_out) for h in self.param_heads], dim=1)
            return other
        else:
            mu_input = torch.cat([cls_out, mu_feats], dim=1)
            mu_pred = self.mu_head(mu_input)
            other = torch.cat([h(cls_out) for h in self.param_heads], dim=1)
            return torch.cat([mu_pred, other], dim=1)


# --- MLP Regressor ---
class MLPRegressor(BaseSVModel):
    """
    MLP regressor for SV parameter estimation.
    Takes the flattened time series as input.
    """
    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.input_dim = config.input_dim # Will be None if not provided in config
        mlp_hidden_dims = config.mlp_hidden_dims

        layers = []
        
        if not mlp_hidden_dims:
            # Default in ModelConfig is [128, 64], so this path is unlikely with default config.
            # Consider raising ValueError or ensuring ModelConfig validates mlp_hidden_dims.
            warnings.warn("MLPRegressor initialized with empty mlp_hidden_dims. Backbone might be trivial.", UserWarning)
            current_hidden_dim = 0 # Placeholder, actual feature_dim will depend on handling of empty mlp_hidden_dims
        else:
            # First layer
            if self.input_dim is None: # Use LazyLinear if input_dim not known
                layers.append(nn.LazyLinear(mlp_hidden_dims[0]))
            else:
                layers.append(nn.Linear(self.input_dim, mlp_hidden_dims[0]))
            
            layers.append(nn.LeakyReLU(1e-2))
            layers.append(nn.LayerNorm(mlp_hidden_dims[0]))
            layers.append(nn.Dropout(config.dropout_rate))

            current_hidden_dim = mlp_hidden_dims[0]

            # Subsequent layers
            for i in range(1, len(mlp_hidden_dims)):
                next_hidden_dim = mlp_hidden_dims[i]
                layers.append(nn.Linear(current_hidden_dim, next_hidden_dim))
                layers.append(nn.LeakyReLU(1e-2))
                layers.append(nn.LayerNorm(next_hidden_dim))
                layers.append(nn.Dropout(config.dropout_rate))
                current_hidden_dim = next_hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.feature_dim = current_hidden_dim

        # Mu head
        mu_feature_dim = 2
        self.mu_head = nn.Sequential(
            nn.Linear(self.feature_dim + mu_feature_dim, self.config.hidden_dim // 2 if self.config.hidden_dim >1 else 1), # Ensure hidden_dim > 1
            nn.LeakyReLU(1e-2),
            nn.LayerNorm(self.config.hidden_dim // 2 if self.config.hidden_dim >1 else 1),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.config.hidden_dim // 2 if self.config.hidden_dim >1 else 1, 1)
        )

        # Parameter heads for other parameters
        self.param_heads = self._create_heads(self.feature_dim, self.n_params - 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mu_mode: str = "raw") -> torch.Tensor:
        original_input = self._handle_input(x) # (B, L) for MLP

        # Flatten the input time series for MLP
        # The _handle_input for MLP already ensures x is (B, L)
        x_flattened = original_input.view(original_input.size(0), -1) # (B, L) -> (B, L)

        features = self.backbone(x_flattened) # (B, feature_dim)

        if mu_mode == "analytical":
            mu_features = self._compute_mu_features(original_input, mu_mode) # (B,1), uses (B,L) input
            other_preds = torch.cat([head(features) for head in self.param_heads], dim=1)
            return other_preds
        else:
            mu_features = self._compute_mu_features(original_input, mu_mode) # (B,2), uses (B,L) input
            mu_input = torch.cat([features, mu_features], dim=1)
            mu_pred = self.mu_head(mu_input)

            other_preds = torch.cat([head(features) for head in self.param_heads], dim=1)
            return torch.cat([mu_pred, other_preds], dim=1)


def validate_model_output(model: nn.Module, sample_input: torch.Tensor,
                         expected_dim: int, mu_mode: str) -> bool:
    """Validate that model output has expected dimensions."""

    model.eval()
    with torch.no_grad():
        try:
            output = model(sample_input, mu_mode)
            actual_dim = output.shape[1]

            if actual_dim != expected_dim:
                warnings.warn(
                    f"Model output dimension mismatch: expected {expected_dim}, "
                    f"got {actual_dim} for mu_mode='{mu_mode}'"
                )
                return False

            return True

        except Exception as e:
            warnings.warn(f"Model validation failed: {e}")
            return False


def build_model(
    kind: str = "cnn",
    model_type: str = "svcj",
    n_params: Optional[int] = None,
    config: Optional[ModelConfig] = None,
    # Added for MLP
    seq_len: Optional[int] = None, # Optional: used if config is None and input_dim not in kwargs. MLP can infer if not provided.
    **kwargs
) -> nn.Module:
    """
    Enhanced factory function to build SV family models.

    Parameters
    ----------
    kind : str, default="cnn"
        Model architecture: "cnn", "cnn_ficura", "tcn", "transformer", "mlp"
    model_type : str, default="svcj"
        SV model type: "sv", "svj", "svcj"
    n_params : int, optional
        Number of parameters. If None, inferred from model_type.
    config : ModelConfig, optional
        Model configuration. If None, created with defaults.
    seq_len: int, optional
        Sequence length, required for MLP if not providing a full config.
    **kwargs
        Additional arguments for ModelConfig.

    Returns
    -------
    nn.Module
        Configured model with proper log_vars parameter.
    """

    # Resolve legacy aliases
    kind = kind.lower()
    if kind == "cnn_svjd":
        kind = "cnn"  # Legacy compatibility

    # Get parameter count
    if n_params is None:
        try:
            from svcj.dl.config import get_param_order
            n_params = len(get_param_order(model_type))
        except ImportError:
            # Fallback parameter counts
            param_counts = {"sv": 4, "svj": 7, "svcj": 7}
            n_params = param_counts.get(model_type.lower(), 7)

    # Create configuration
    if config is None:
        # MLP needs input_dim (seq_len) to be passed to config
        # Use a copy of kwargs to avoid modifying the original dict
        current_build_kwargs = kwargs.copy()
        if kind == "mlp":
            if 'input_dim' not in current_build_kwargs and seq_len is not None:
                current_build_kwargs['input_dim'] = seq_len
            # No error if input_dim is still None for MLP; ModelConfig will have input_dim=None
            # Removed ValueError for MLP:
            # elif 'input_dim' not in kwargs and seq_len is None:
            #     raise ValueError("seq_len (as input_dim) must be provided for MLP when config is None.")
        elif kind == "transformer": # Transformer needs input_dim or uses default
            if 'input_dim' not in current_build_kwargs and seq_len is not None:
                current_build_kwargs['input_dim'] = seq_len
            # If still None, TransformerRegressor handles it with a default max_len

        config = ModelConfig(
            n_params=n_params,
            model_type=model_type,
            arch_type=kind,
            **current_build_kwargs # Pass potentially modified kwargs
        )

    # Build model based on architecture
    if kind == "cnn":
        model = CNNRegressor(config)
    elif kind == "cnn_ficura":
        model = CNN_Ficura(config)
    elif kind == "tcn":
        model = TCNRegressor(config)
    elif kind == "transformer":
        model = TransformerRegressor(config)
    elif kind == "mlp":
        # Removed check, as MLPRegressor now handles config.input_dim being None by using LazyLinear.
        # if config.input_dim is None: # Should have been set above or in provided config
        #     raise ValueError("input_dim (sequence length) must be specified in ModelConfig for MLP.")
        model = MLPRegressor(config)
    else:
        raise ValueError(f"Unknown model architecture: {kind}")

    # Add heteroscedastic loss parameters if requested
    if config.use_heteroscedastic_loss:
        # log_vars should match the maximum possible output dimension
        # (for analytical mode, it will be automatically subset)
        model.log_vars = nn.Parameter(torch.zeros(n_params))

        # Add method to get effective log_vars
        def get_effective_log_vars(mu_mode: str = "raw") -> torch.Tensor:
            if mu_mode == "analytical":
                return model.log_vars[1:]  # Exclude mu variance
            else:
                return model.log_vars

        model.get_effective_log_vars = get_effective_log_vars

    return model


# ───────────────────────────────────────────────────────────────────────────
# Model utilities and testing
# ---------------------------------------------------------------------------

def test_model_consistency(model: nn.Module, seq_len: int = 252,
                          batch_size: int = 4) -> Dict[str, Any]:
    """Test model consistency across different mu_modes with better error handling."""

    model.eval()
    # MLP expects (B,L), others expect (B,1,L) initially
    if isinstance(model, MLPRegressor):
        sample_input = torch.randn(batch_size, seq_len)
    else:
        sample_input = torch.randn(batch_size, 1, seq_len)


    results = {}

    for mu_mode in ["raw", "analytical", "devol"]:
        try:
            with torch.no_grad():
                output = model(sample_input, mu_mode)
                results[mu_mode] = {
                    'shape': output.shape,
                    'dtype': output.dtype,
                    'mean': float(output.mean()),
                    'std': float(output.std()),
                    'min': float(output.min()),
                    'max': float(output.max()),
                }
        except Exception as e:
            import traceback
            results[mu_mode] = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    return results


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count by module type
    param_by_type = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0:
                module_type = type(module).__name__
                param_by_type[module_type] = param_by_type.get(module_type, 0) + module_params

    return {
        'total': total_params,
        'trainable': trainable_params,
        'by_type': param_by_type
    }


# Example usage and testing
if __name__ == "__main__":
    # Test model creation and consistency
    n_params_for_test = 7
    seq_len_for_test = 252

    for arch in ["cnn", "cnn_ficura", "tcn", "transformer", "mlp"]:
        print(f"\nTesting {arch} architecture:")

        try:
            # For MLP, input_dim (seq_len) is crucial
            if arch == "mlp":
                config = ModelConfig(n_params=n_params_for_test, model_type="svcj", arch_type=arch, input_dim=seq_len_for_test)
            else:
                config = ModelConfig(n_params=n_params_for_test, model_type="svcj", arch_type=arch)

            model = build_model(arch, config=config)

            # Count parameters
            param_info = count_parameters(model)
            print(f"  Parameters: {param_info['total']:,} total, {param_info['trainable']:,} trainable")

            # Test consistency
            consistency = test_model_consistency(model, seq_len=seq_len_for_test)
            for mode, result in consistency.items():
                if 'error' in result:
                    print(f"  {mode}: ERROR - {result['error']}")
                else:
                    print(f"  {mode}: shape={result['shape']}, range=[{result['min']:.3f}, {result['max']:.3f}]")

        except Exception as e:
            import traceback
            print(f"  ERROR building/testing {arch}: {e}")
            # print(traceback.format_exc())