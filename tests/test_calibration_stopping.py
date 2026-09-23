import unittest
import numpy as np
from src.calibration_stopping import observed_features, select_budget, fit_policy, tune_policy, decisions, FEATURES, Policy, outcome_key
from src.calibration_orders import calibration_ids, ORDERS, subset_code
from unittest.mock import patch


class StoppingTests(unittest.TestCase):
    def test_orders_separate_episodes_but_not_participant_folds(self):
        rows=[]; outcomes={}
        for p in range(4):
            for order in ORDERS:
                for b in (1,2):
                    rows.append({'participant':p,'session':2,'order':order,
                                 **{k:float(b if k=='budget' else p/10+b/20) for k in FEATURES}})
                for b in (1,2,3):
                    outcomes[p,2,order,b]=.7+b*.01
        self.assertEqual(len(decisions(Policy('fixed',1),rows)),24)
        original_fit=fit_policy
        seen=[]
        def checked_fit(train,y,*args,**kwargs):
            people={r['participant'] for r in train}
            for p in people:
                self.assertEqual({r['order'] for r in train if r['participant']==p},set(ORDERS))
            self.assertEqual({k[0] for k in y},people)
            seen.append(len(people))
            return original_fit(train,y,*args,**kwargs)
        with patch('src.calibration_stopping.fit_policy',side_effect=checked_fit):
            tune_policy(rows,outcomes,'pre_error')
        self.assertEqual(set(seen),{3,4})

    def test_round_permutations_preserve_pool_and_scoring(self):
        rows=[{'participant':1,'session':2,'group':'development','role':'calibration','calibration_rank':rank,'class_index':label}
              for label in range(17) for rank in (1,2,3)]
        rows.append({'participant':1,'session':2,'group':'development','role':'scoring','calibration_rank':1,'class_index':0})
        for order in ORDERS:
            ranks=list(map(int,order))
            one=calibration_ids(rows,1,2,ranks[:1]);two=calibration_ids(rows,1,2,ranks[:2]);full=calibration_ids(rows,1,2,ranks)
            self.assertTrue(set(one)<set(two)<set(full))
            self.assertEqual(set(full),set(range(51)))
        self.assertEqual(subset_code((3,1)),subset_code((1,3)))
        with self.assertRaises(ValueError):
            calibration_ids(rows,1,2,(1,1))
    def test_no_future_observation_after_stop(self):
        visited = []
        def observe(b):
            visited.append(b)
            if b == 2:
                raise AssertionError('Future round was read')
            return {'budget': b}
        budget, trace = select_budget(lambda r: (True, 0), observe)
        self.assertEqual((budget, visited), (1, [1]))

    def test_whole_balanced_trials_required(self):
        probs = np.tile([[.8, .2], [.1, .9]], (35, 1))
        labels = np.repeat([0, 1], 35)
        features = observed_features(probs, probs, labels, 1)
        self.assertEqual(features['probability_change'], 0)
        with self.assertRaises(ValueError):
            observed_features(probs, probs, np.tile([0, 1], 35), 1)
        with self.assertRaises(ValueError):
            observed_features(probs, probs, np.zeros(70, dtype=int), 1)

    def test_outer_outcomes_cannot_affect_training_or_choices(self):
        rows = []
        outcomes = {}
        for p in range(5):
            for b in (1, 2):
                rows.append({'participant': p, 'session': 2,
                             **{k: float(b if k == 'budget' else p / 10 + b / 20) for k in FEATURES}})
            for b in (1, 2, 3):
                outcomes[p, 2, b] = .6 + b * .02 + p * .01
        train = [r for r in rows if r['participant'] != 4]
        train_y = {k: v for k, v in outcomes.items() if k[0] != 4}
        test = [r for r in rows if r['participant'] == 4]
        one, spec1 = tune_policy(train, train_y, 'ridge')
        for k in outcomes:
            if k[0] == 4:
                outcomes[k] = -1000
        two, spec2 = tune_policy(train, {k:v for k,v in outcomes.items() if k[0] != 4}, 'ridge')
        self.assertEqual(spec1, spec2)
        self.assertEqual(decisions(one, test), decisions(two, test))
        with self.assertRaises(ValueError):
            tune_policy(train, outcomes, 'ridge')
        self.assertEqual(one.model.named_steps['standardscaler'].n_samples_seen_, len(train))


if __name__ == '__main__':
    unittest.main()
