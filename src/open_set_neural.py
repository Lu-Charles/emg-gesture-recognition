"""Known-only CNN and equation-based PredIN adaptations; no new algorithm claim."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.emg_cnn import CompactEMGNet


def remap_known(y):
    y=np.asarray(y,dtype=np.int64)
    if not len(y) or np.any((y<10)|(y>16)):
        raise ValueError('Unknown classes cannot enter model fitting')
    return y-10


class Branch(nn.Module):
    def __init__(self, prototype):
        super().__init__()
        self.encoder=CompactEMGNet(classes=7).encoder
        self.embedding=nn.Linear(64,128)
        self.prototype=prototype
        if prototype:self.prototypes=nn.Parameter(torch.randn(7,128))
        else:self.head=nn.Linear(128,7)

    def forward(self,x):
        z=self.embedding(self.encoder(x))
        scores=z@self.prototypes.T if self.prototype else self.head(z)
        return z,scores


def compactness(z,prototypes,y):
    """Literal manuscript v2 Eq5, not silently replaced by smooth-L1.

    Branch condition is the vector L1 norm; low branch is half L2 norm
    (not squared), high branch L1 minus0.5. This unusual convention is disclosed.
    """
    delta=z-prototypes[y]
    l1=delta.abs().sum(1);l2=torch.linalg.vector_norm(delta,dim=1)
    return torch.where(l1<1,.5*l2,l1-.5).mean()


def triplet(z,prototypes,y):
    distances=torch.linalg.vector_norm(z[:,None]-prototypes[None],dim=2)
    mask=F.one_hot(y,7).bool()
    nearest=distances.masked_fill(mask,float('inf')).min(1).values
    positive=distances.gather(1,y[:,None]).squeeze(1)
    return (positive-nearest+1.).clamp_min(0).mean()


def inconsistency(logits_a,logits_b,y):
    negative=~F.one_hot(y,7).bool()
    def distribution(logits):
        positive=logits.gather(1,y[:,None]).detach()
        # Eqs8,10: softmax(-d), d=-max(pos-neg-m1,0).
        margin=(positive-logits-.5).clamp_min(0)
        return margin[negative].reshape(-1,6).softmax(1)
    pa,pb=distribution(logits_a),distribution(logits_b)
    mass=(pa*(1-pb)+pb*(1-pa)).sum(1)
    return -mass.clamp_min(1e-12).log().mean()


class OpenSetNet(nn.Module):
    def __init__(self,family,seed=42):
        super().__init__()
        if family not in ('cnn','pl','predin'):raise ValueError('Unsupported family')
        self.family=family
        branches=[]
        for b in range(2 if family=='predin' else 1):
            torch.manual_seed(seed+b)
            branches.append(Branch(prototype=family!='cnn'))
        self.branches=nn.ModuleList(branches)

    def forward(self,x):
        return [branch(x) for branch in self.branches]

    def loss(self,x,y):
        output=self(x);terms={}
        ce=torch.stack([F.cross_entropy(logits,y) for _,logits in output]).sum()
        terms['ce']=ce
        if self.family!='cnn':
            terms['compactness']=torch.stack([compactness(z,b.prototypes,y) for b,(z,_) in zip(self.branches,output)]).sum()
        if self.family=='predin':
            terms['triplet']=torch.stack([triplet(z,b.prototypes,y) for b,(z,_) in zip(self.branches,output)]).sum()
            terms['inconsistency']=inconsistency(output[0][1],output[1][1],y)
        # All published weights1; branch losses sum, shared inconsistency once.
        return sum(terms.values()),terms

    def predict_scores(self,x):
        output=self(x)
        averaged=torch.stack([s for _,s in output]).mean(0)
        pred=averaged.argmax(1)+10
        scores={'probability':averaged.softmax(1).max(1).values}
        if self.family!='cnn':scores['prototype']=averaged.max(1).values
        return pred,scores


def known_moments(corpus,trial_ids):
    ids=np.asarray(trial_ids,dtype=np.int64)
    remap_known(corpus.labels[ids])
    if any(corpus.rows[i]['group']!='train' for i in ids):
        raise ValueError('Normalization must fit train participants only')
    total=np.zeros(16);squares=np.zeros(16);count=0
    for offset in range(0,len(ids),16):
        x,_=corpus.batch(corpus.window_ids(ids[offset:offset+16]))
        x=x.astype(np.float64);total+=x.sum(axis=(0,2));squares+=(x*x).sum(axis=(0,2));count+=x.shape[0]*x.shape[2]
    mean=total/count;scale=np.sqrt(np.maximum(squares/count-mean*mean,1e-16))
    return mean.reshape(1,16,1),scale.reshape(1,16,1),count


def infer(model,x,device):
    model.eval();pred=[];scores={}
    with torch.no_grad():
        for offset in range(0,len(x),128):
            p,s=model.predict_scores(torch.from_numpy(x[offset:offset+128]).to(device))
            pred.append(p.cpu().numpy())
            for key,value in s.items():scores.setdefault(key,[]).append(value.cpu().numpy())
    result={k:np.concatenate(v).astype(float) for k,v in scores.items()}
    if not all(np.isfinite(v).all() for v in result.values()):raise FloatingPointError('Nonfinite inference')
    return np.concatenate(pred),result
