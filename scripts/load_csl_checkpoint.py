"""Load a study checkpoint with its saved forward-variant metadata."""
import json
from pathlib import Path
from scripts.verify_csl_author_recipe import restore

def load_checkpoint(path):
    path=Path(path)
    if path.suffix!='.pt':raise ValueError('Expected a .pt checkpoint with adjacent _state.json metadata')
    metadata=json.loads((path.parent/(path.stem+'_state.json')).read_text())
    variant=metadata.get('forward_variant')
    if variant not in (None,'resample_then_gain_then_bn'):raise ValueError(f'Unrecognized forward variant: {variant}')
    model=restore(path.stem,path.parent)
    if variant=='resample_then_gain_then_bn':
        from scripts.csl_input_order import attach
        model=attach(model)
    return model.eval()
