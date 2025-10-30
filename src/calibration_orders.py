"""Whole-round order changes inside the frozen calibration pool only."""
from itertools import permutations

ORDERS = tuple(''.join(map(str, order)) for order in permutations((1, 2, 3)))


def subset_code(ranks):
    ranks = tuple(ranks)
    if len(set(ranks)) != len(ranks) or not set(ranks) <= {1, 2, 3}:
        raise ValueError('Invalid calibration ranks')
    return ''.join(map(str, sorted(ranks))) or '0'


def calibration_ids(rows, participant, session, ranks):
    subset_code(ranks)
    selected = [i for i,r in enumerate(rows) if int(r['participant'])==participant
                and int(r['session'])==session and r['role']=='calibration'
                and int(r['calibration_rank']) in ranks]
    if any(rows[i]['group']!='development' for i in selected):
        raise ValueError('Development participants only')
    if len(selected)!=17*len(ranks):
        raise ValueError('Incomplete balanced calibration subset')
    for rank in ranks:
        labels=[int(rows[i]['class_index']) for i in selected if int(rows[i]['calibration_rank'])==rank]
        if sorted(labels)!=list(range(17)):
            raise ValueError('Each round must contain one whole trial per class')
    return selected
