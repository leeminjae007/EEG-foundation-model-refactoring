import json

import pytest

from scripts.test_tuab_interim_candidates import validation_best


def test_validation_best_uses_bacc_and_earliest_tie():
    records = [dict(epoch=1, balanced_accuracy=.7),
               dict(epoch=2, balanced_accuracy=.8),
               dict(epoch=3, balanced_accuracy=.8)]
    raw = ('\n'.join(json.dumps(row) for row in records) + '\n').encode()
    assert validation_best(raw)['epoch'] == 2


def test_validation_best_rejects_nonfinite():
    with pytest.raises(ValueError):
        validation_best(b'{"epoch":1,"balanced_accuracy":NaN}\n')
