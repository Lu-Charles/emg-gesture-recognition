"""Check unchanged author SAL components using synthetic inputs, not EMG results."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit', type=Path, default=Path('research/runs/20260907_sal_code_audit'))
    a = ap.parse_args(); started = time.perf_counter()
    snapshot = a.audit / 'paper_era_snapshot'
    ledger = json.loads((a.audit / 'paper_era_snapshot_sha256.json').read_text())
    for name, expected in ledger['files'].items():
        assert hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == expected
    sys.path.insert(0, str(snapshot.resolve()))
    nu = importlib.import_module('networks_utils')
    nets = importlib.import_module('networks')
    tensors = importlib.import_module('tensorize_emg')
    torch.set_num_threads(1); torch.manual_seed(42)
    checks = []
    def passed(name, **evidence): checks.append(dict(name=name, passed=True, **evidence))
    layer = nu.SpatialAdaptation((7,24))
    x = torch.randn(3,1,7,24)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        out = layer(x)
    error = float((out-x).abs().max())
    assert error < 2e-5
    passed('identity_on_original_CSL_grid', max_absolute_error=error)
    # The source convention is a backward sampling grid, so +1 sample shifts content left.
    with torch.no_grad(): layer.xshift.fill_(2/24)
    ramp = torch.arange(24).float().reshape(1,1,1,24).expand(1,1,7,24)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        shifted = layer(ramp)
    torch.testing.assert_close(shifted[:,:,:,:-1], ramp[:,:,:,1:], rtol=1e-5, atol=2e-5)
    passed('translation_scale_and_direction', normalized_shift=2/24, pixel_sampling_shift=1)
    layer = nu.SpatialAdaptation((7,24)).double()
    with torch.no_grad():
        for p,v in zip(layer.parameters(),[.023,-.017,.031,1.012,.983,.009,-.011]): p.fill_(v)
    x = torch.randn(2,1,7,24,dtype=torch.float64); weights = torch.randn_like(x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        loss = (layer(x)*weights).sum(); loss.backward()
        differences = {}
        for name,p in layer.named_parameters():
            g=float(p.grad); v=float(p.detach()); h=1e-7
            with torch.no_grad():
                p.fill_(v+h); hi=float((layer(x)*weights).sum())
                p.fill_(v-h); lo=float((layer(x)*weights).sum()); p.fill_(v)
            fd=(hi-lo)/(2*h); differences[name]=dict(gradient=g,finite_difference=fd)
            assert np.isfinite(g) and abs(g-fd) <= 1e-5 + 1e-5*abs(fd), (name,g,fd)
    passed('seven_affine_parameter_gradients', derivatives=differences)
    # Exercise the author's segmented split function with trial identities embedded in values.
    obj=tensors.EMGSegmentData.__new__(tensors.EMGSegmentData)
    obj.intrasession=False; obj.num_repetitions=10
    S,G,R,T=2,3,10,4
    ids=torch.arange(S*G*R).reshape(S,G,R)
    obj.X=ids[:,:,:,None,None,None,None].expand(S,G,R,T,1,1,1).float().clone()
    obj.Y=torch.arange(G)[None,:,None,None].expand(S,G,R,T).clone()
    obj.active=np.ones((S,G,R,T),dtype=bool)
    obj.durations=np.full((S,G,R),T)
    full=obj.get_tensors(train_session=0,test_session=1,rep_idx=0)
    partial=obj.get_tensors(train_session=0,test_session=1,rep_idx=0,gest_idxs=[1])
    source,cal,score=[set(full[i].reshape(-1).tolist()) for i in (0,2,4)]
    assert source==set(range(30)) and cal=={30,40,50} and score==set(range(30,60))-cal
    assert not source&cal and not source&score and not cal&score
    torch.testing.assert_close(full[4],partial[4]); torch.testing.assert_close(full[5],partial[5])
    assert set(partial[2].reshape(-1).tolist())=={40}
    assert sum(full[6])==len(full[5])
    passed('whole_trial_split_and_identical_scoring', source_trials=len(source), calibration_trials=len(cal), scoring_trials=len(score))
    model=nets.LogisticRegressor(num_classes=3,input_shape=(7,24),channels=168,p_input=.5)
    model.eval()
    for p in model.parameters(): p.requires_grad=False
    for p in model.spatial_adapt.parameters(): p.requires_grad=True
    model.baseline.requires_grad=True
    old={n:v.clone() for n,v in model.state_dict().items()}
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
    model.input_dropout.train()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        opt.zero_grad(); loss=torch.nn.functional.cross_entropy(model(torch.randn(20,1,7,24)),torch.arange(20)%3); loss.backward();opt.step()
    changed=[n for n,v in model.state_dict().items() if not torch.equal(v,old[n])]
    assert changed and all(n=='baseline' or n.startswith('spatial_adapt.') for n in changed)
    assert torch.equal(model.bn.running_mean,old['bn.running_mean'])
    passed('adaptation_preserves_classifier_and_BN_statistics', changed_state=changed)
    model.eval(); x=torch.randn(10,1,7,24)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        torch.testing.assert_close(model(x),model(x),rtol=0,atol=0)
    passed('deterministic_scoring_in_eval_mode')
    # Preserve observable implementation issues; do not silently patch author snapshots.
    obj=tensors.EMGData(dataset='capgmyo',input_shape=(1,1),fs=2,sessions=['s'],num_gestures=1,num_repetitions=1,remove_baseline=True)
    observations=dict(remove_baseline_argument_preserved=obj.remove_baseline,
        original_runner_enables_input_dropout_during_adaptation=True,
        original_runner_lacks_explicit_eval_before_adapted_scoring=True,
        missing_repetitions_are_oversampled_before_split_in_original_loader=True,
        note='Source observations, not evidence about published numerical results; local runner must explicitly score in eval mode and reject duplicate trial leakage.')
    result=dict(scope='SYNTHETIC COMPONENT AUDIT ONLY; no dataset or classification experiment',revision=ledger['revision'],passed=True,
                checks=checks,observations=observations,wall_seconds=time.perf_counter()-started,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.audit/'component_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
