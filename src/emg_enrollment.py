"""A continuous enrollment trajectory with isolated evaluation checkpoints."""
import copy
import time
import numpy as np
import torch
from scripts.pilot_emg_cnn import sync


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def enrollment_path(model, x, y, checkpoints, lr, seed, device):
    """Yield CPU snapshots without resetting Adam or the permutation generator.

    Consumers may evaluate/mutate snapshots without changing the trajectory.
    Training timings exclude time the generator is suspended for evaluation.
    """
    if not checkpoints or checkpoints != sorted(set(checkpoints)) or checkpoints[0] < 1:
        raise ValueError("Checkpoints must be increasing unique positive epochs")
    xt = torch.from_numpy(x).to(device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = np.random.default_rng(seed)
    losses, seconds = [], 0.
    model.train()
    for epoch in range(1, checkpoints[-1] + 1):
        sync(device)
        tick = time.perf_counter()
        order = rng.permutation(len(x))
        total = 0.
        for j in range(0, len(order), 128):
            ids = torch.tensor(order[j:j + 128], dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(xt[ids]), yt[ids])
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(ids)
        sync(device)
        seconds += time.perf_counter() - tick
        losses.append(total / len(x))
        if not np.isfinite(losses[-1]):
            raise ValueError("Nonfinite enrollment loss")
        if epoch in checkpoints:
            yield {"epoch": epoch, "model": cpu_copy(model.state_dict()),
                   "optimizer": cpu_copy(optimizer.state_dict()),
                   "epoch_losses": list(losses), "fit_seconds": seconds}
