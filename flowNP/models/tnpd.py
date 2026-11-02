import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from attrdictionary import AttrDict

from models.tnp import TNP


class TNPD(TNP):
    ''' TNP-D (Nguyen & Grover, 2022): same encoder as TNPA, but all targets are decoded
    in one pass with a diagonal Gaussian (no autoregression). Added for the glucose
    project; not part of the original flowNP repo. '''
    def __init__(
        self,
        dim_x,
        dim_y,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers,
        sigma_bound=0.0,
    ):
        super(TNPD, self).__init__(
            dim_x,
            dim_y,
            d_model,
            emb_depth,
            dim_feedforward,
            nhead,
            dropout,
            num_layers,
            sigma_bound
        )

        self.predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_y*2)
        )

    def predict(self, xc, yc, xt, num_samples=None):
        batch = AttrDict(xc=xc, yc=yc, xt=xt, yt=torch.zeros(*xt.shape[:-1], yc.shape[-1], device=xt.device))
        z_target = self.encode(batch, autoreg=False)
        mean, std = torch.chunk(self.predictor(z_target), 2, dim=-1)
        if self.sigma_bound > 0.0:
            std = self.sigma_bound + (1-self.sigma_bound) * F.softplus(std)
        else:
            std = torch.exp(std)
        return Normal(mean, std)

    def forward(self, batch, num_samples=None, reduce_ll=True):
        pred_tar = self.predict(batch.xc, batch.yc, batch.xt)
        outs = AttrDict()
        if reduce_ll:
            outs.tar_ll = pred_tar.log_prob(batch.yt).sum(-1).mean()
        else:
            outs.tar_ll = pred_tar.log_prob(batch.yt).sum(-1)
        outs.loss = - (outs.tar_ll)
        return outs
