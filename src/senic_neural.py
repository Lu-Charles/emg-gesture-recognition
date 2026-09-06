"""Compact amplitude-preserving CNN and circular-channel augmentation for SeNic.

The permutation operation follows ABSDA's principle; this is an eight-channel
temporal-backbone adaptation, not reproduction of its HD-EMG spatial CNN.
"""
import numpy as np
import torch
from torch import nn


class SeNicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv1d(8,32,7,stride=2,padding=3),nn.ReLU(),
            nn.Conv1d(32,64,5,stride=2,padding=2),nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.head=nn.Linear(72,7)

    def forward(self,x):
        # Explicit per-channel amplitude survives alongside learned time features.
        amplitude=torch.log1p(x.abs().mean(-1))
        return self.head(torch.cat([self.encoder(x),amplitude],dim=1))


def augment_channels(x,shifts):
    if x.ndim!=3 or x.shape[1]!=8 or shifts.shape!=(len(x),):
        raise ValueError('Expected windows x8 x time and one integer shift per window')
    indices=(torch.arange(8,device=x.device)[None,:]-shifts[:,None])%8
    return x.gather(1,indices[:,:,None].expand(-1,-1,x.shape[2]))


def training_scale(raw_trials,ids):
    if not ids:raise ValueError('No normalization training trials')
    x=raw_trials[ids].astype(np.float64)
    if x.ndim!=3 or x.shape[1:]!=(400,8) or not np.isfinite(x).all():
        raise ValueError('Expected finite unique samples: trials x400 x8')
    return float(max(np.sqrt(np.mean(x*x)),1e-8))


def probabilities(model,x):
    model.eval()
    with torch.no_grad():return model(torch.as_tensor(x,dtype=torch.float32)).softmax(1).numpy()


def fit(model,x,y,steps,seed,shifts=(0,),lr=.001):
    """Fixed minibatch Adam steps; no scoring-based stopping or checkpoint choice."""
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed);aug_rng=np.random.default_rng(seed+10000)
    xt=torch.as_tensor(x,dtype=torch.float32);yt=torch.as_tensor(y,dtype=torch.long)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    losses=[];model.train()
    for _ in range(steps):
        ids=torch.tensor(rng.choice(len(x),size=min(64,len(x)),replace=False))
        data=xt[ids]
        if len(shifts)>1:data=augment_channels(data,torch.tensor(aug_rng.choice(shifts,len(ids)),dtype=torch.long))
        opt.zero_grad();loss=nn.functional.cross_entropy(model(data),yt[ids])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        loss.backward();opt.step();losses.append(float(loss.detach()))
    return dict(losses=losses,training_accuracy=float(np.mean(probabilities(model,x).argmax(1)==y)))
