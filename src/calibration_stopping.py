"""Small acquisition-stopping policies. No scoring labels enter decisions."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ('budget', 'pre_error', 'pre_nll', 'pre_trial_nll_std',
            'post_entropy', 'post_error', 'prediction_change', 'probability_change')


def episode_key(row):
    """Orders identify replay episodes, never independent participants."""
    return (row['participant'], row['session']) + ((row['order'],) if 'order' in row else ())


def outcome_key(row, budget):
    return episode_key(row) + (int(budget),)


def observed_features(before, after, labels, budget, windows_per_trial=35):
    """Only the newly acquired balanced round, before/after its update.

    Post-update quantities are fitted-data diagnostics, not validation estimates.
    The previous checkpoint never saw this round. Trials have equal length.
    """
    before, after = np.asarray(before), np.asarray(after)
    labels = np.asarray(labels)
    if before.shape != after.shape or before.ndim != 2 or len(labels) != len(before):
        raise ValueError('Probability/label shape mismatch')
    if budget not in (1, 2) or len(labels) % windows_per_trial:
        raise ValueError('Invalid round or incomplete trial')
    for p in (before, after):
        if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5):
            raise ValueError('Invalid probabilities')
    trial_labels = labels.reshape(-1, windows_per_trial)
    if not np.all(trial_labels == trial_labels[:, :1]):
        raise ValueError('Trial crosses labels')
    if sorted(trial_labels[:, 0].tolist()) != list(range(before.shape[1])):
        raise ValueError('Expected exactly one whole trial per class')
    nll = -np.log(np.maximum(before[np.arange(len(labels)), labels], 1e-12))
    return dict(zip(FEATURES, map(float, (
        budget, np.mean(before.argmax(1) != labels), nll.mean(),
        nll.reshape(-1, windows_per_trial).mean(1).std(),
        -(after * np.log(np.maximum(after, 1e-12))).sum(1).mean(),
        np.mean(after.argmax(1) != labels),
        np.mean(before.argmax(1) != after.argmax(1)),
        np.abs(after - before).sum(1).mean() / 2,
    ))))


def select_budget(decide, observe):
    """A stopped policy cannot request the next round's features."""
    trace = []
    for budget in (1, 2):
        record = observe(budget)
        if int(record['budget']) != budget:
            raise ValueError('Observation belongs to wrong round')
        stop, value = decide(record)
        trace.append({'budget': budget, 'stop': bool(stop), 'value': float(value)})
        if stop:
            return budget, trace
    return 3, trace


@dataclass
class Policy:
    family: str
    parameter: float
    alpha: float = 0.
    model: object = None
    thresholds: dict = None

    def decide(self, record):
        if self.family == 'fixed':
            return record['budget'] >= self.parameter, record['budget']
        if self.family == 'ridge':
            value = float(self.model.predict([[record[k] for k in FEATURES]])[0])
            return value <= self.parameter, value
        value = float(record[self.family])
        threshold = self.thresholds[int(record['budget'])]
        return value <= threshold, value


def fit_policy(rows, outcomes, family, parameter, alpha=0.):
    """Call with training participants only; outcomes needed only for ridge."""
    policy = Policy(family, float(parameter), float(alpha))
    if family == 'fixed':
        return policy
    if family == 'ridge':
        x = [[r[k] for k in FEATURES] for r in rows]
        # Average remaining benefit per additional round, not next-round benefit.
        y = [(outcomes[outcome_key(r, 3)] -
              outcomes[outcome_key(r, r['budget'])]) /
             (3 - r['budget']) for r in rows]
        policy.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x, y)
    elif family in ('pre_error', 'post_entropy', 'probability_change'):
        policy.thresholds = {}
        for budget in (1, 2):
            values = [r[family] for r in rows if r['budget'] == budget]
            policy.thresholds[budget] = (float('-inf') if parameter < 0 else
                                         float('inf') if parameter > 1 else
                                         float(np.quantile(values, parameter)))
    else:
        raise ValueError('Unknown policy')
    return policy


def decisions(policy, rows):
    indexed = {outcome_key(r, r['budget']): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError('Duplicate decision state')
    result = []
    for key in sorted({episode_key(r) for r in rows}):
        budget, trace = select_budget(policy.decide, lambda b: indexed[key + (b,)])
        identity = {'participant': key[0], 'session': key[1]}
        if len(key) == 3:
            identity['order'] = key[2]
        result.append({**identity, 'selected_budget': budget, 'trace': trace})
    return result


def assess(chosen, outcomes):
    f1 = np.array([outcomes[outcome_key(r, r['selected_budget'])] for r in chosen])
    full = np.array([outcomes[outcome_key(r, 3)] for r in chosen])
    budget = np.array([r['selected_budget'] for r in chosen])
    return {'mean_macro_f1': float(f1.mean()), 'mean_seconds': float(budget.mean() * 85),
            'mean_loss_vs_full': float((full - f1).mean()),
            'worst_loss_vs_full': float((full - f1).max()),
            'fraction_loss_over_002': float(((full - f1) > .02).mean())}


def candidates(family):
    if family == 'ridge':
        return [(t, a) for a in (1., 10., 100.) for t in (-1., 0., .0025, .005, .01, .02, .04, .08, 1.)]
    if family == 'fixed':
        return [(1., 0.), (2., 0.), (3., 0.)]
    return [(q, 0.) for q in (-1., 0., .1, .25, .5, .75, .9, 1., 2.)]


def tune_policy(rows, outcomes, family):
    """Inner participant CV; screen mean F1 loss <= .01, then minimize time.

    This screen is not a guarantee for an unseen participant. Outer evaluation
    must remain separate. Stable candidate ordering resolves exact ties.
    """
    people = sorted({r['participant'] for r in rows})
    if len(people) < 3:
        raise ValueError('Need at least three training participants')
    if {key[0] for key in outcomes} != set(people):
        raise ValueError('Outcomes outside training participants')
    table = []
    for parameter, alpha in candidates(family):
        chosen = []
        for held in people:
            train = [r for r in rows if r['participant'] != held]
            score = [r for r in rows if r['participant'] == held]
            train_outcomes = {k: v for k, v in outcomes.items() if k[0] != held}
            policy = fit_policy(train, train_outcomes, family, parameter, alpha)
            chosen.extend(decisions(policy, score))
        table.append({'parameter': parameter, 'alpha': alpha, **assess(chosen, outcomes)})
    feasible = [r for r in table if r['mean_loss_vs_full'] <= .01 + 1e-12]
    if not feasible:
        # No empirical fit to the loss constraint: explicit full-budget fallback.
        return Policy('fixed', 3), {'fallback': True, 'candidates': table}
    selected = min(feasible, key=lambda r: (r['mean_seconds'], r['mean_loss_vs_full']))
    policy = fit_policy(rows, outcomes, family, selected['parameter'], selected['alpha'])
    return policy, {'fallback': False, 'selected': selected, 'candidates': table}
