import json
from argparse import Namespace
import torch
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm
import logging
from typing import List, Optional, Dict

import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class FairseqDropout(nn.Module):
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and (self.training or self.apply_during_inference):
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

    def make_generation_fast_(
            self,
            name: str,
            retain_dropout: bool = False,
            retain_dropout_modules: Optional[List[str]] = None,
            **kwargs
    ):
        if retain_dropout:
            if retain_dropout_modules is not None and self.module_name is None:
                logger.warning(
                    "Cannot enable dropout during inference for module {} "
                    "because module_name was not set".format(name)
                )
            elif (
                    retain_dropout_modules is None  # if None, apply to all modules
                    or self.module_name in retain_dropout_modules
            ):
                logger.info(
                    "Enabling dropout during inference for module: {}".format(name)
                )
                self.apply_during_inference = True
            else:
                logger.info("Disabling dropout for module: {}".format(name))


class VariancePredictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                args.encoder_embed_dim,
                args.var_pred_hidden_dim,
                kernel_size=args.var_pred_kernel_size,
                padding=(args.var_pred_kernel_size - 1) // 2,
            ),
            nn.ReLU(),
        )
        self.ln1 = nn.LayerNorm(args.var_pred_hidden_dim)
        self.dropout_module = FairseqDropout(
            p=args.var_pred_dropout, module_name=self.__class__.__name__
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                args.var_pred_hidden_dim,
                args.var_pred_hidden_dim,
                kernel_size=args.var_pred_kernel_size,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.ln2 = nn.LayerNorm(args.var_pred_hidden_dim)
        self.proj = nn.Linear(args.var_pred_hidden_dim, 1)

    def forward(self, x):
        # Input: B x T x C; Output: B x T
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout_module(self.ln1(x))
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout_module(self.ln2(x))
        return self.proj(x).squeeze(dim=2)


class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class Generator(torch.nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.num_kernels = len(cfg["resblock_kernel_sizes"])
        self.num_upsamples = len(cfg["upsample_rates"])
        self.conv_pre = weight_norm(
            Conv1d(
                cfg.get("model_in_dim", 80),
                cfg["upsample_initial_channel"],
                7,
                1,
                padding=3,
            )
        )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
                zip(cfg["upsample_rates"], cfg["upsample_kernel_sizes"])
        ):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        cfg["upsample_initial_channel"] // (2 ** i),
                        cfg["upsample_initial_channel"] // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = cfg["upsample_initial_channel"] // (2 ** (i + 1))
            for k, d in zip(
                    cfg["resblock_kernel_sizes"], cfg["resblock_dilation_sizes"]
            ):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        # Removing weight norm...
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class CodeHiFiGANModel(Generator):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dict = nn.Embedding(cfg["num_embeddings"], cfg["embedding_dim"])
        self.multispkr = cfg.get("multispkr", None)
        self.embedder = cfg.get("embedder_params", None)

        if self.multispkr and not self.embedder:
            self.spkr = nn.Embedding(cfg.get("num_speakers", 200), cfg["embedding_dim"])
        elif self.embedder:
            self.spkr = nn.Linear(cfg.get("embedder_dim", 256), cfg["embedding_dim"])

        self.dur_predictor = None
        if cfg.get("dur_predictor_params", None):
            self.dur_predictor = VariancePredictor(
                Namespace(**cfg["dur_predictor_params"])
            )

        self.f0 = cfg.get("f0", None)
        n_f0_bin = cfg.get("f0_quant_num_bin", 0)
        self.f0_quant_embed = (
            None if n_f0_bin <= 0 else nn.Embedding(n_f0_bin, cfg["embedding_dim"])
        )

    @staticmethod
    def _upsample(signal, max_frames):
        if signal.dim() == 3:
            bsz, channels, cond_length = signal.size()
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
            bsz, channels, cond_length = signal.size()
        else:
            signal = signal.view(-1, 1, 1)
            bsz, channels, cond_length = signal.size()

        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)

        # pad zeros as needed (if signal's shape does not divide completely with max_frames)
        reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if reminder > 0:
            raise NotImplementedError(
                "Padding condition signal - misalignment between condition features."
            )

        signal = signal.view(bsz, channels, max_frames)
        return signal

    def forward(self, **kwargs):
        x = self.dict(kwargs["code"]).transpose(1, 2)

        if self.dur_predictor and kwargs.get("dur_prediction", False):
            assert x.size(0) == 1, "only support single sample"
            log_dur_pred = self.dur_predictor(x.transpose(1, 2))
            dur_out = torch.clamp(
                torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1
            )
            # B x C x T
            x = torch.repeat_interleave(x, dur_out.view(-1), dim=2)

        if self.f0:
            if self.f0_quant_embed:
                kwargs["f0"] = self.f0_quant_embed(kwargs["f0"].long()).transpose(1, 2)
            else:
                kwargs["f0"] = kwargs["f0"].unsqueeze(1)

            if x.shape[-1] < kwargs["f0"].shape[-1]:
                x = self._upsample(x, kwargs["f0"].shape[-1])
            elif x.shape[-1] > kwargs["f0"].shape[-1]:
                kwargs["f0"] = self._upsample(kwargs["f0"], x.shape[-1])
            x = torch.cat([x, kwargs["f0"]], dim=1)

        if self.multispkr:
            assert (
                    "spkr" in kwargs
            ), 'require "spkr" input for multispeaker CodeHiFiGAN vocoder'
            spkr = self.spkr(kwargs["spkr"]).transpose(1, 2)
            spkr = self._upsample(spkr, x.shape[-1])
            x = torch.cat([x, spkr], dim=1)

        for k, feat in kwargs.items():
            if k in ["spkr", "code", "f0", "dur_prediction"]:
                continue

            feat = self._upsample(feat, x.shape[-1])
            x = torch.cat([x, feat], dim=1)

        return super().forward(x)


class CodeHiFiGANVocoder(nn.Module):
    def __init__(
            self, checkpoint_path: str, model_cfg=None, fp16: bool = False
    ) -> None:
        super().__init__()
        if not model_cfg:
            default_vocoder_cfg = {
                "resblock": "1",
                "batch_size": 16,
                "learning_rate": 0.0002,
                "adam_b1": 0.8,
                "adam_b2": 0.99,
                "lr_decay": 0.999,
                "seed": 1234,

                "upsample_rates": [5, 4, 4, 2, 2],
                "upsample_kernel_sizes": [11, 8, 8, 4, 4],
                "upsample_initial_channel": 512,
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "num_embeddings": 1000,
                "embedding_dim": 128,
                "model_in_dim": 128,

                "segment_size": 8960,
                "code_hop_size": 320,
                "f0": False,
                "num_mels": 80,
                "num_freq": 1025,
                "n_fft": 1024,
                "hop_size": 256,
                "win_size": 1024,

                "dur_prediction_weight": 1.0,
                "dur_predictor_params": {
                    "encoder_embed_dim": 128,
                    "var_pred_hidden_dim": 128,
                    "var_pred_kernel_size": 3,
                    "var_pred_dropout": 0.5
                },

                "sampling_rate": 16000,

                "fmin": 0,
                "fmax": 8000,
                "fmax_for_loss": None,

                "num_workers": 4,

                "dist_config": {
                    "dist_backend": "nccl",
                    "dist_url": "env://"
                }
            }
            model_cfg = default_vocoder_cfg
        self.model = CodeHiFiGANModel(model_cfg)
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict["generator"])
        self.model.eval()
        if fp16:
            self.model.half()
        self.model.remove_weight_norm()
        logger.info(f"loaded CodeHiFiGAN checkpoint from {checkpoint_path}")

    def forward(self, x: Dict[str, torch.Tensor], dur_prediction=False) -> torch.Tensor:
        assert "code" in x
        x["dur_prediction"] = dur_prediction
        # remove invalid code
        mask = x["code"] >= 0
        x["code"] = x["code"][mask].unsqueeze(dim=0)
        if "f0" in x:
            f0_up_ratio = x["f0"].size(1) // x["code"].size(1)
            mask = mask.unsqueeze(2).repeat(1, 1, f0_up_ratio).view(-1, x["f0"].size(1))
            x["f0"] = x["f0"][mask].unsqueeze(dim=0)

        return self.model(**x).detach().squeeze()

    @classmethod
    def from_data_cfg(cls, args, data_cfg):
        vocoder_cfg = data_cfg.vocoder
        assert vocoder_cfg is not None, "vocoder not specified in the data config"
        with open(vocoder_cfg["config"]) as f:
            model_cfg = json.load(f)
        return cls(vocoder_cfg["checkpoint"], model_cfg, fp16=args.fp16)
