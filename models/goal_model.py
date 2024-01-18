import math
from copy import deepcopy
from typing import Callable, Dict, Literal, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


def resize_image(img, target_resolution=(128, 128)):
    if type(img) == np.ndarray:
        img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
    elif type(img) == torch.Tensor:
        img = F.interpolate(img, size=target_resolution, mode="bilinear")
    else:
        raise ValueError
    return img


def get_activation(activation: str | Callable | None) -> Callable:
    if not activation:
        return nn.Identity
    elif callable(activation):
        return activation
    ACT_LAYER = {
        "tanh": nn.Tanh,
        "relu": lambda: nn.ReLU(inplace=True),
        "leaky_relu": lambda: nn.LeakyReLU(inplace=True),
        "swish": lambda: nn.SiLU(inplace=True),  # SiLU is alias for Swish
        "sigmoid": nn.Sigmoid,
        "elu": lambda: nn.ELU(inplace=True),
        "gelu": nn.GELU,
    }
    activation = activation.lower()
    assert activation in ACT_LAYER, f"Supported activations: {ACT_LAYER.keys()}"
    return ACT_LAYER[activation]


def discrete_horizon(horizon):
    """
    0 - 10: 0
    10 - 20: 1
    20 - 30: 2
    30 - 40: 3
    40 - 50: 4
    50 - 60: 5
    60 - 70: 6
    70 - 80: 7
    80 - 90: 8
    90 - 100: 9
    100 - 120: 10
    120 - 140: 11
    140 - 160: 12
    160 - 180: 13
    180 - 200: 14
    200 - ...: 15
    """
    # horizon_list = [0]*25 + [1]*25 + [2]*25 + [3]*25 +[4]* 50 + [5]*50 + [6] * 700
    horizon_list = []
    for i in range(10):
        horizon_list += [i] * 10
    for i in range(10, 15):
        horizon_list += [i] * 20
    horizon_list += [15] * 700
    if type(horizon) == torch.Tensor:
        return torch.Tensor(horizon_list, device=horizon.device)[horizon]
    elif type(horizon) == np.ndarray:
        return np.array(horizon_list)[horizon]
    elif type(horizon) == int:
        return horizon_list[horizon]
    else:
        assert False


class Concat(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(2 * input_dim, output_dim)

    def forward(self, model_x, model_y):
        return self.fc(torch.cat([model_x, model_y], dim=-1))


class Bilinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.U = nn.Linear(input_dim, output_dim)
        self.V = nn.Linear(input_dim, output_dim)
        self.P = nn.Linear(input_dim, output_dim)

    def forward(self, model_x, model_y):
        return self.P(torch.tanh(self.U(model_x) * self.V(model_y)))


class FiLM(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.U = nn.Linear(input_dim, output_dim)
        self.V = nn.Linear(input_dim, output_dim)

    def forward(self, model_x, model_y):
        return self.U(model_y) * model_x + self.V(model_y)


class PrevActionEmbedding(nn.Module):
    def __init__(self, output_dim: int, action_space):
        super().__init__()
        self.output_dim = output_dim
        self.action_space = action_space
        embed_dim = output_dim // len(action_space)
        self._embed = nn.ModuleList(
            [nn.Embedding(voc_size, embed_dim) for voc_size in self.action_space]
        )
        self._fc = nn.Linear(len(self.action_space) * embed_dim, output_dim)

    def forward(self, prev_action):
        categorical = prev_action.shape[-1]
        action_list = []
        for i in range(categorical):
            action_list.append(self._embed[i](prev_action[..., i].long()))
        output = self._fc(torch.cat(action_list, dim=-1))
        return output


class ExtraObsEmbedding(nn.Module):
    def __init__(self, embed_dims: dict, output_dim: int):
        super().__init__()
        self.embed_dims = embed_dims
        self.embed_biome = nn.Embedding(168, embed_dims["biome_hiddim"])
        self.embed_compass = build_mlp(
            input_dim=2,
            hidden_dim=embed_dims["compass_hiddim"],
            output_dim=embed_dims["compass_hiddim"],
            hidden_depth=2,
        )
        self.embed_gps = build_mlp(
            input_dim=3,
            hidden_dim=embed_dims["gps_hiddim"],
            output_dim=embed_dims["gps_hiddim"],
            hidden_depth=2,
        )
        self.embed_voxels = nn.Embedding(32, embed_dims["voxels_hiddim"] // 4)
        self.embed_voxels_last = build_mlp(
            input_dim=12 * embed_dims["voxels_hiddim"] // 4,
            hidden_dim=embed_dims["voxels_hiddim"],
            output_dim=embed_dims["voxels_hiddim"],
            hidden_depth=2,
        )
        sum_dims = sum(v for v in embed_dims.values())
        self.fusion = build_mlp(
            input_dim=sum_dims,
            hidden_dim=sum_dims // 2,
            output_dim=output_dim,
            hidden_depth=2,
        )

    def forward(self, obs: dict):
        biome = obs["biome"]
        compass = obs["compass"]
        gps = obs["gps"]
        voxels = obs["voxels"]

        with_time_dimension = len(biome.shape) == 2

        if with_time_dimension:
            B, T = biome.shape
            biome = biome.view(B * T, *biome.shape[2:])
            compass = compass.view(B * T, *compass.shape[2:])
            gps = gps.view(B * T, *gps.shape[2:])
            voxels = voxels.view(B * T, *voxels.shape[2:])

        biome = self.embed_biome(biome)
        compass = self.embed_compass(compass)
        gps = self.embed_gps(gps)
        voxels = self.embed_voxels(voxels)
        voxels = self.embed_voxels_last(voxels.view(voxels.shape[0], -1))

        output = self.fusion(torch.cat([biome, compass, gps, voxels], dim=-1))

        if with_time_dimension:
            output = output.view(B, T, *output.shape[1:])

        return output


def build_mlp(
    input_dim,
    *,
    hidden_dim: int,
    output_dim: int,
    hidden_depth: int = None,
    num_layers: int = None,
    activation: str | Callable = "relu",
    norm_type: Literal["batchnorm", "layernorm"] | None = None,
    add_input_activation: bool | str | Callable = False,
    add_input_norm: bool = False,
    add_output_activation: bool | str | Callable = False,
    add_output_norm: bool = False,
) -> nn.Sequential:
    """
    In other popular RL implementations, tanh is typically used with orthogonal
    initialization, which may perform better than ReLU.

    Args:
        norm_type: None, "batchnorm", "layernorm", applied to intermediate layers
        add_input_activation: whether to add a nonlinearity to the input _before_
            the MLP computation. This is useful for processing a feature from a preceding
            image encoder, for example. Image encoder typically has a linear layer
            at the end, and we don't want the MLP to immediately stack another linear
            layer on the input features.
            - True to add the same activation as the rest of the MLP
            - str to add an activation of a different type.
        add_input_norm: see `add_input_activation`, whether to add a normalization layer
            to the input _before_ the MLP computation.
            values: True to add the `norm_type` to the input
        add_output_activation: whether to add a nonlinearity to the output _after_ the
            MLP computation.
            - True to add the same activation as the rest of the MLP
            - str to add an activation of a different type.
        add_output_norm: see `add_output_activation`, whether to add a normalization layer
            _after_ the MLP computation.
            values: True to add the `norm_type` to the input
    """
    assert (hidden_depth is None) != (num_layers is None), (
        "Either hidden_depth or num_layers must be specified, but not both. "
        "num_layers is defined as hidden_depth+1"
    )
    if hidden_depth is not None:
        assert hidden_depth >= 0
    if num_layers is not None:
        assert num_layers >= 1
    act_layer = get_activation(activation)

    if norm_type is not None:
        norm_type = norm_type.lower()

    if not norm_type:
        norm_type = nn.Identity
    elif norm_type == "batchnorm":
        norm_type = nn.BatchNorm1d
    elif norm_type == "layernorm":
        norm_type = nn.LayerNorm
    else:
        raise ValueError(f"Unsupported norm layer: {norm_type}")

    hidden_depth = num_layers - 1 if hidden_depth is None else hidden_depth
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), norm_type(hidden_dim), act_layer()]
        for i in range(hidden_depth - 1):
            mods += [
                nn.Linear(hidden_dim, hidden_dim),
                norm_type(hidden_dim),
                act_layer(),
            ]
        mods.append(nn.Linear(hidden_dim, output_dim))

    if add_input_norm:
        mods = [norm_type(input_dim)] + mods
    if add_input_activation:
        if add_input_activation is not True:
            act_layer = get_activation(add_input_activation)
        mods = [act_layer()] + mods
    if add_output_norm:
        mods.append(norm_type(output_dim))
    if add_output_activation:
        if add_output_activation is not True:
            act_layer = get_activation(add_output_activation)
        mods.append(act_layer())

    return nn.Sequential(*mods)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        *,
        hidden_dim: int,
        output_dim: int,
        hidden_depth: int = None,
        num_layers: int = None,
        activation: str | Callable = "relu",
        weight_init: str | Callable = "orthogonal",
        bias_init="zeros",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
        add_input_activation: bool | str | Callable = False,
        add_input_norm: bool = False,
        add_output_activation: bool | str | Callable = False,
        add_output_norm: bool = False,
    ):
        super().__init__()
        # delegate to build_mlp by keywords
        self.layers = build_mlp(
            input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            hidden_depth=hidden_depth,
            num_layers=num_layers,
            activation=activation,
            weight_init=weight_init,
            bias_init=bias_init,
            norm_type=norm_type,
            add_input_activation=add_input_activation,
            add_input_norm=add_input_norm,
            add_output_activation=add_output_activation,
            add_output_norm=add_output_norm,
        )
        # add attributes to the class
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_depth = hidden_depth
        self.activation = activation
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.norm_type = norm_type
        if add_input_activation is True:
            self.input_activation = activation
        else:
            self.input_activation = add_input_activation
        if add_input_norm is True:
            self.input_norm_type = norm_type
        else:
            self.input_norm_type = None
        # do the same for output activation and norm
        if add_output_activation is True:
            self.output_activation = activation
        else:
            self.output_activation = add_output_activation
        if add_output_norm is True:
            self.output_norm_type = norm_type
        else:
            self.output_norm_type = None

    def forward(self, x):
        return self.layers(x)


class SimpleNetwork(nn.Module):
    def __init__(
        self,
        action_space: list,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        num_cat: int,
        hidden_size: int,
        fusion_type: str,
        max_ep_len: int = 4096,
        backbone: nn.Module = None,
        frozen_cnn: bool = False,
        use_recurrent: str = None,
        use_extra_obs: bool = False,
        use_horizon: bool = False,
        use_pred_horizon: bool = False,
        use_prev_action: bool = False,
        extra_obs_cfg: dict = None,
        c: int = 1,
        **kwargs,
    ):
        super().__init__()

        types = ["rgb", "concat", "bilinear", "film"]

        assert fusion_type in types, f"ERROR: [{fusion_type}] is not in {types}"

        self.action_space = action_space
        self.num_cat = num_cat
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.fusion_type = fusion_type
        self.frozen_cnn = frozen_cnn
        self.hidden_size = hidden_size
        self.act_pred_dim = sum(self.action_space)
        self.use_recurrent = use_recurrent
        self.use_extra_obs = use_extra_obs
        self.use_horizon = use_horizon
        self.use_prev_action = use_prev_action
        self.use_pred_horizon = use_pred_horizon
        self.c = c

        assert (
            not use_pred_horizon
        ) or use_horizon, "use_pred_horizon is based on use_horizon!"

        self.backbone = backbone
        if frozen_cnn:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

        self.extra_dim = 0

        # self.embed_goal = nn.Embedding(num_cat, hidden_size)
        self.embed_goal = nn.Linear(self.goal_dim, hidden_size)
        self.embed_rgb = nn.Linear(self.state_dim, hidden_size)

        if self.use_extra_obs:
            assert (
                extra_obs_cfg is not None
            ), "ExtraObsEmbedding class arguments are required!"
            self.embed_extra = ExtraObsEmbedding(
                embed_dims=extra_obs_cfg, output_dim=hidden_size
            )
            self.extra_dim += hidden_size

        if self.use_prev_action:
            self.embed_prev_action = PrevActionEmbedding(
                output_dim=hidden_size, action_space=self.action_space
            )
            self.extra_dim += hidden_size

        if fusion_type == "rgb":
            concat_input_dim = hidden_size + self.extra_dim
        elif fusion_type in ["concat"]:
            concat_input_dim = hidden_size * 2 + self.extra_dim
        elif fusion_type == "bilinear":
            concat_input_dim = hidden_size + self.extra_dim
            self.f_rgb_goal = Bilinear(hidden_size, hidden_size)
        elif fusion_type == "film":
            concat_input_dim = hidden_size + self.extra_dim
            self.f_rgb_goal = FiLM(hidden_size, hidden_size)
        else:
            assert False

        self.concat_input = build_mlp(
            input_dim=concat_input_dim,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            hidden_depth=3,
        )

        if self.use_recurrent == "gru":
            self.recurrent = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0.1,
            )
        elif self.use_recurrent == "transformer":
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
            self.embed_ln = nn.LayerNorm(hidden_size)
            transformer_cfg = kwargs["transformer_cfg"]
            config = transformers.GPT2Config(
                vocab_size=1,  # doesn't matter -- we don't use the vocab
                n_embd=hidden_size,
                n_layer=transformer_cfg["n_layer"],
                n_head=transformer_cfg["n_head"],
                n_inner=transformer_cfg["n_head"] * hidden_size,
                activation_function=transformer_cfg["activation_function"],
                resid_pdrop=transformer_cfg["resid_pdrop"],
                attn_pdrop=transformer_cfg["attn_pdrop"],
            )
            self.recurrent = transformers.GPT2Model(config)

        if self.use_horizon:
            self.embed_horizon = nn.Embedding(16, hidden_size)
            self.fuse_horizon = Concat(hidden_size, hidden_size)

        if self.use_pred_horizon:
            self.pred_horizon = build_mlp(
                input_dim=hidden_size,
                hidden_dim=hidden_size,
                output_dim=16,
                hidden_depth=2,
            )

        action_modules = [
            build_mlp(
                input_dim=hidden_size,
                hidden_dim=hidden_size,
                output_dim=self.act_pred_dim,
                hidden_depth=2,
            )
        ]

        self.action_head = nn.Sequential(*action_modules)

    def _img_feature(self, img, goal_embeddings):
        """
        do the normalization inside the backbone
        """
        if img.shape[-1] == 3:
            img = img.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = img.shape
        img = img.view(B * T, *img.shape[2:])
        goal_embeddings = goal_embeddings.view(B * T, *goal_embeddings.shape[2:])
        feat = self.backbone(img, goal_embeddings)
        feat = feat.view(B, T, -1)
        return feat

    def forward(self, goals, states, horizons, timesteps=None, attention_mask=None):
        mid_info = {}
        raw_rgb = states["rgb"]
        batch_size, seq_length = raw_rgb.shape[:2]
        assert (
            self.use_recurrent or seq_length == 1
        ), "simple network only supports length = 1 if use_recurrent = None. "
        goal_embeddings = self.embed_goal(goals)
        rgb_embeddings = self.embed_rgb(self._img_feature(raw_rgb, goal_embeddings))

        #! compute rgb embeddings based on goal information
        if self.fusion_type == "rgb":
            body_embeddings = rgb_embeddings
        elif self.fusion_type == "concat":
            body_embeddings = torch.cat([rgb_embeddings, goal_embeddings], dim=-1)
        elif self.fusion_type == "bilinear":
            body_embeddings = self.f_rgb_goal(rgb_embeddings, goal_embeddings)
        elif self.fusion_type == "film":
            body_embeddings = self.f_rgb_goal(rgb_embeddings, goal_embeddings)
        elif self.fusion_type == "multihead":
            body_embeddings = rgb_embeddings
        else:
            assert False, "unknown fusion type. "

        #! add extra observation embeddings
        if self.use_extra_obs:
            extra_obs = states
            extra_embeddings = self.embed_extra(extra_obs)
            body_embeddings = torch.cat([body_embeddings, extra_embeddings], dim=-1)

        #! add prev action embeddings
        if self.use_prev_action:
            prev_action_embeddings = self.embed_prev_action(states["prev_action"])
            body_embeddings = torch.cat(
                [body_embeddings, prev_action_embeddings], dim=-1
            )

        obs_feature = self.concat_input(body_embeddings)

        #! recurrent network is used to
        if self.use_recurrent in ["gru", "lstm"]:
            obs_feature, hids = self.recurrent(obs_feature)
        elif self.use_recurrent in ["transformer"]:
            time_embeddings = self.embed_timestep(timesteps)
            inputs_embeds = obs_feature + time_embeddings
            # inputs_embeds = self.embed_ln(obs_feature + time_embeddings)
            transformer_outputs = self.recurrent(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            obs_feature = transformer_outputs["last_hidden_state"]

        #! add horizon embeddings
        if self.use_horizon:
            if self.use_pred_horizon:
                pred_horizons = self.pred_horizon(obs_feature)
                mid_info["pred_horizons"] = pred_horizons
                if not self.training:
                    mid_horizons = pred_horizons.argmax(-1)
                    mid_horizons = (mid_horizons - self.c).clip(0)
                else:
                    mid_horizons = horizons
            else:
                mid_horizons = horizons
            horizon_embeddings = self.embed_horizon(mid_horizons)
            mid_feature = self.fuse_horizon(obs_feature, horizon_embeddings)
        else:
            mid_feature = obs_feature

        action_preds = self.action_head(mid_feature)

        return action_preds, mid_info

    def get_action(self, goals, states, horizons):
        # augment the batch dimension
        goals = goals.unsqueeze(0)  # 1xLxH
        B, L, _ = goals.shape
        for k, v in states.items():
            states[k] = v.unsqueeze(0)
        if horizons is not None:
            horizons = horizons.unsqueeze(0)  # 1xL
        timesteps = torch.arange(L).unsqueeze(0).to(goals.device)
        attention_mask = torch.ones((B, L)).to(goals.device)
        return self.forward(goals, states, horizons, timesteps, attention_mask)


class FanInInitReLULayer(nn.Module):
    """Implements a slightly modified init that correctly produces std 1 outputs given ReLU activation
    :param inchan: number of input channels
    :param outchan: number of output channels
    :param layer_args: positional layer args
    :param layer_type: options are "linear" (dense layer), "conv" (2D Convolution), "conv3d" (3D convolution)
    :param init_scale: multiplier on initial weights
    :param batch_norm: use batch norm after the layer (for 2D data)
    :param group_norm_groups: if not None, use group norm with this many groups after the layer. Group norm 1
        would be equivalent of layernorm for 2D data.
    :param layer_norm: use layernorm after the layer (for 1D data)
    :param layer_kwargs: keyword arguments for the layer
    """

    def __init__(
        self,
        inchan: int,
        outchan: int,
        *layer_args,
        layer_type: str = "conv",
        init_scale: int = 1,
        batch_norm: bool = False,
        batch_norm_kwargs: Dict = {},
        group_norm_groups: Optional[int] = None,
        layer_norm: bool = False,
        use_activation=True,
        log_scope: Optional[str] = None,
        **layer_kwargs,
    ):
        super().__init__()

        # Normalization
        self.norm = None
        if batch_norm:
            self.norm = nn.BatchNorm2d(inchan, **batch_norm_kwargs)
        elif group_norm_groups is not None:
            self.norm = nn.GroupNorm(group_norm_groups, inchan)
        elif layer_norm:
            self.norm = nn.LayerNorm(inchan)

        layer = dict(conv=nn.Conv2d, conv3d=nn.Conv3d, linear=nn.Linear)[layer_type]
        self.layer = layer(
            inchan, outchan, bias=self.norm is None, *layer_args, **layer_kwargs
        )

        # Init Weights (Fan-In)
        self.layer.weight.data *= init_scale / self.layer.weight.norm(
            dim=tuple(range(1, self.layer.weight.data.ndim)), p=2, keepdim=True
        )
        # Init Bias
        if self.layer.bias is not None:
            self.layer.bias.data *= 0

    def forward(self, x):
        """Norm after the activation. Experimented with this for both IAM and BC and it was slightly better."""
        if self.norm is not None:
            x = self.norm(x)
        x = self.layer(x)
        if self.use_activation:
            x = F.relu(x, inplace=True)
        return x

    def get_log_keys(self):
        return [
            f"activation_mean/{self.log_scope}",
            f"activation_std/{self.log_scope}",
        ]


class CnnBasicBlock(nn.Module):
    """
    Residual basic block, as in ImpalaCNN. Preserves channel number and shape
    :param inchan: number of input channels
    :param init_scale: weight init scale multiplier
    """

    def __init__(
        self,
        inchan: int,
        init_scale: float = 1,
        log_scope="",
        init_norm_kwargs: Dict = {},
        **kwargs,
    ):
        super().__init__()
        self.inchan = inchan
        self.goal_dim = kwargs["goal_dim"]
        s = math.sqrt(init_scale)
        self.conv0 = FanInInitReLULayer(
            self.inchan,
            self.inchan,
            kernel_size=3,
            padding=1,
            init_scale=s,
            log_scope=f"{log_scope}/conv0",
            **init_norm_kwargs,
        )
        self.conv1 = FanInInitReLULayer(
            self.inchan,
            self.inchan,
            kernel_size=3,
            padding=1,
            init_scale=s,
            log_scope=f"{log_scope}/conv1",
            **init_norm_kwargs,
        )
        self.goal_gate = nn.Sequential(
            nn.Linear(self.goal_dim, self.inchan * 2),
            nn.ReLU(),
            nn.Linear(self.inchan * 2, self.inchan),
        )

    def forward(self, x, goal_embeddings):
        px = self.conv1(self.conv0(x))
        gate = self.goal_gate(goal_embeddings)
        # print(gate.shape, px.shape)
        r = x + px * gate.sigmoid().unsqueeze(2).unsqueeze(3)
        return r


class CnnDownStack(nn.Module):
    """
    Downsampling stack from Impala CNN.
    :param inchan: number of input channels
    :param nblock: number of residual blocks after downsampling
    :param outchan: number of output channels
    :param init_scale: weight init scale multiplier
    :param pool: if true, downsample with max pool
    :param post_pool_groups: if not None, normalize with group norm with this many groups
    :param kwargs: remaining kwargs are passed into the blocks and layers
    """

    name = "Impala_CnnDownStack"

    def __init__(
        self,
        inchan: int,
        nblock: int,
        outchan: int,
        init_scale: float = 1,
        pool: bool = True,
        post_pool_groups: Optional[int] = None,
        log_scope: str = "",
        init_norm_kwargs: Dict = {},
        first_conv_norm=False,
        **kwargs,
    ):
        super().__init__()
        self.inchan = inchan
        self.outchan = outchan
        self.pool = pool
        first_conv_init_kwargs = deepcopy(init_norm_kwargs)
        if not first_conv_norm:
            first_conv_init_kwargs["group_norm_groups"] = None
            first_conv_init_kwargs["batch_norm"] = False
        self.firstconv = FanInInitReLULayer(
            inchan,
            outchan,
            kernel_size=3,
            padding=1,
            log_scope=f"{log_scope}/firstconv",
            **first_conv_init_kwargs,
        )
        self.post_pool_groups = post_pool_groups
        if post_pool_groups is not None:
            self.n = nn.GroupNorm(post_pool_groups, outchan)
        self.blocks = nn.ModuleList(
            [
                CnnBasicBlock(
                    outchan,
                    init_scale=init_scale / math.sqrt(nblock),
                    log_scope=f"{log_scope}/block{i}",
                    init_norm_kwargs=init_norm_kwargs,
                    **kwargs,
                )
                for i in range(nblock)
            ]
        )

    def forward(self, x, *args):
        x = self.firstconv(x)
        if self.pool:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            if self.post_pool_groups is not None:
                x = self.n(x)

        # sequential
        for layer in self.blocks:
            x = layer(x, *args)
        return x

    def output_shape(self, inshape):
        c, h, w = inshape
        assert c == self.inchan
        if self.pool:
            return (self.outchan, (h + 1) // 2, (w + 1) // 2)
        else:
            return (self.outchan, h, w)


class GoalImpalaCNN(nn.Module):
    """
    :param inshape: input image shape (height, width, channels)
    :param chans: number of residual downsample stacks. Each element is the number of
        filters per convolution in the stack
    :param outsize: output hidden size
    :param nblock: number of residual blocks per stack. Each block has 2 convs and a residual
    :param init_norm_kwargs: arguments to be passed to convolutional layers. Options can be found
        in ypt.model.util:FanInInitReLULayer
    :param dense_init_norm_kwargs: arguments to be passed to convolutional layers. Options can be found
        in ypt.model.util:FanInInitReLULayer
    :param kwargs: remaining kwargs are passed into the CnnDownStacks
    """

    name = "GoalImpalaCNN"

    def __init__(
        self,
        inshape: list[int],
        chans: list[int],
        outsize: int,
        nblock: int,
        init_norm_kwargs: dict = {},
        dense_init_norm_kwargs: dict = {},
        first_conv_norm=False,
        **kwargs,
    ):
        super().__init__()
        h, w, c = inshape
        curshape = (c, h, w)
        self.stacks = nn.ModuleList()
        for i, outchan in enumerate(chans):
            stack = CnnDownStack(
                curshape[0],
                nblock=nblock,
                outchan=outchan,
                init_scale=math.sqrt(len(chans)),
                log_scope=f"downstack{i}",
                init_norm_kwargs=init_norm_kwargs,
                first_conv_norm=first_conv_norm if i == 0 else True,
                **kwargs,
            )
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)

        c1, c2, c3 = curshape

        self.dense = FanInInitReLULayer(
            c1 * c2 * c3,
            outsize,
            layer_type="linear",
            log_scope="imapala_final_dense",
            init_scale=1.4,
            **dense_init_norm_kwargs,
        )
        self.outsize = outsize

    def forward(self, x, *args):
        for layer in self.stacks:
            x = layer(x, *args)
        x = x.reshape((x.shape[0], -1))
        x = self.dense(x)
        return x


class GoalImpalaCNNWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        net_config = {
            "hidsize": 1024,
            "img_shape": [128, 128, 3],
            "impala_chans": [16, 32, 32],
            "impala_kwargs": {"post_pool_groups": 1, "goal_dim": 1024},
            "impala_width": 4,
            "init_norm_kwargs": {"batch_norm": False, "group_norm_groups": 1},
        }
        hidsize = net_config["hidsize"]
        img_shape = net_config["img_shape"]
        impala_width = net_config["impala_width"]
        impala_chans = net_config["impala_chans"]
        impala_kwargs = net_config["impala_kwargs"]
        init_norm_kwargs = net_config["init_norm_kwargs"]

        chans = tuple(int(impala_width * c) for c in impala_chans)
        self.dense_init_norm_kwargs = deepcopy(init_norm_kwargs)
        if self.dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            self.dense_init_norm_kwargs.pop("group_norm_groups", None)
            self.dense_init_norm_kwargs["layer_norm"] = True
        if self.dense_init_norm_kwargs.get("batch_norm", False):
            self.dense_init_norm_kwargs.pop("batch_norm", False)
            self.dense_init_norm_kwargs["layer_norm"] = True

        self.cnn = GoalImpalaCNN(
            outsize=256,
            output_size=hidsize,
            inshape=img_shape,
            chans=chans,
            nblock=2,
            init_norm_kwargs=init_norm_kwargs,
            dense_init_norm_kwargs=self.dense_init_norm_kwargs,
            first_conv_norm=False,
            **impala_kwargs,
            # **kwargs,
        )

        self.linear = FanInInitReLULayer(
            256,
            hidsize,
            layer_type="linear",
            **self.dense_init_norm_kwargs,
        )

    def forward(self, img, goal_embeddings):
        """
        img: BxT, 3, H, W, without normalization
        goal_embeddings: BxT, C
        """
        img = resize_image(img, (128, 128))
        img = img.to(dtype=torch.float32) / 255.0
        return self.linear(self.cnn(img, goal_embeddings))

    def get_cam_layer(self):
        return [block.conv1 for block in self.cnn.stacks[-1].blocks] + [
            block.conv0 for block in self.cnn.stacks[-1].blocks
        ]


def get_goal_model(cfg, action_space):
    print(
        dict(
            action_space=action_space,
            state_dim=1024,
            goal_dim=512,
            action_dim=8,
            num_cat=len(cfg["data"]["filters"]),
            hidden_size=1024,
            fusion_type="concat",
            max_ep_len=1000,
            frozen_cnn=False,
            use_recurrent="transformer",
            use_extra_obs=True,
            use_horizon=True,
            use_prev_action=True,
            extra_obs_cfg={
                "biome_hiddim": 256,
                "compass_hiddim": 256,
                "gps_hiddim": 256,
                "voxels_hiddim": 256,
            },
            use_pred_horizon=True,
            c=8,
            transformer_cfg={
                "n_layer": 6,
                "n_head": 4,
                "resid_pdrop": 0.1,
                "attn_pdrop": 0.1,
                "activation_function": "relu",
            },
        )
    )
    model = SimpleNetwork(
        action_space=action_space,
        state_dim=1024,
        goal_dim=512,
        action_dim=8,
        num_cat=len(cfg["data"]["filters"]),
        hidden_size=1024,
        fusion_type="concat",
        max_ep_len=1000,
        backbone=GoalImpalaCNNWrapper(),
        frozen_cnn=False,
        use_recurrent="transformer",
        use_extra_obs=True,
        use_horizon=True,
        use_prev_action=True,
        extra_obs_cfg={
            "biome_hiddim": 256,
            "compass_hiddim": 256,
            "gps_hiddim": 256,
            "voxels_hiddim": 256,
        },
        use_pred_horizon=True,
        c=8,
        transformer_cfg={
            "n_layer": 6,
            "n_head": 4,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "activation_function": "relu",
        },
    )
    state_dict = torch.load(cfg["model"]["load_ckpt_path"], map_location="cpu")
    print(f"[MAIN] load checkpoint from {cfg['model']['load_ckpt_path']}. ")
    model.load_state_dict(state_dict["model_state_dict"], strict=False)
    return model
