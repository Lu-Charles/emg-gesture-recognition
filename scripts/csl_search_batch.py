"""Batched candidate evaluation through unchanged author forward using torch.func."""
import torch,scipy.stats
from scripts import csl_author_recipe as r
NAMES=('xshift','yshift','rot_theta','xscale','yscale','xshear','yshear')
def candidates(n=16384):
    a=2*torch.tensor(scipy.stats.qmc.LatinHypercube(d=7,seed=42).random(n=n)).float()-1
    a[:,0]=8*a[:,0]/23;a[:,1]=8*a[:,1]/6;a[:,2]=(15/180)*a[:,2]
    a[:,3:5]=torch.pow(1+torch.abs(a[:,3:5])*.1,torch.sign(a[:,3:5]));a[:,5:]*=.1
    return torch.cat([a,torch.tensor([[0.,0.,0.,1.,1.,0.,0.]])])
def losses(model,prototype,labels,a):
    assert not model.training;model.input_transform.constrain_params=False;out=[]
    def forward(params):return torch.func.functional_call(model,params,(prototype,))
    with torch.no_grad():
        for batch in a.split(512):
            params={f'input_transform.{name}.0':batch[:,j] for j,name in enumerate(NAMES)}
            logits=torch.vmap(forward)(params)
            loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),labels.repeat(len(batch)),reduction='none').reshape(len(batch),-1).mean(1)
            out.append(loss)
    model.input_transform.constrain_params=True
    return torch.cat(out)
def search(model,prototype,labels):
    a=candidates();v=losses(model,prototype,labels,a);best=int(v.argmin())
    p=r.load_initial_functions()['get_inv_constrained_params'](*a[best],boundaries=r.BOUNDS)
    with torch.no_grad():
        for name,value in zip(NAMES,p):getattr(model.input_transform,name)[0].copy_(value)
    return dict(index=best,loss=float(v[best]),zero_loss_candidates=int((v==0).sum()),near_min_candidates=int((v<=v.min()+1e-5).sum()))
def check():
    import time,json
    from copy import deepcopy
    from scripts.verify_csl_author_recipe import restore
    from torch.utils.data import DataLoader,TensorDataset
    m=restore('source');m.adaptation_phase=True;m.input_transform.mode='bicubic'
    manifest=json.loads((r.OUT/'manifest.json').read_text());cx,cy,_=r.role_data(manifest,'calibration',21);sx,sy,_=r.role_data(manifest,'source')
    proto=(cx.square().mean(0,keepdim=True)).sqrt();checks=[]
    for gain in (False,True):
        model=deepcopy(m)
        if gain:model.get_session_means(sx[sy==5],cx)
        a=candidates();t=time.perf_counter();v=losses(model,proto,cy[:1],a);batchsec=time.perf_counter()-t
        # Independently evaluate every candidate using literal author forward in a scalar loop.
        reference=[];model.input_transform.constrain_params=False;t=time.perf_counter()
        with torch.no_grad():
            for row in a:
                for name,value in zip(NAMES,row):getattr(model.input_transform,name)[0].copy_(value)
                reference.append(torch.nn.functional.cross_entropy(model(proto),cy[:1]))
        ref=torch.stack(reference);error=float((v-ref).abs().max());torch.testing.assert_close(v,ref,rtol=2e-5,atol=2e-5)
        assert v.argmin()==ref.argmin(),(gain,int(v.argmin()),int(ref.argmin()))
        checks.append(dict(gain=gain,candidates=len(a),max_loss_error=error,matching_argmin=int(v.argmin()),batch_seconds=batchsec,scalar_seconds=time.perf_counter()-t))
    r.dump(r.OUT/'batched_search_checks.json',dict(passed=True,checks=checks));print(checks,flush=True)
if __name__=='__main__':torch.set_num_threads(1);check()
