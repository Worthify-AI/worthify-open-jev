import copy
from pathlib import Path

import pytest

from benchmarks.evaluate_authored_robustness import evaluate, read
from openjev_phase1.core import validate_row


def fixture_predictions():
    rows = read(Path('examples/worthify-robustness.jsonl'))
    outputs = [dict(id=row['id'], option_ids=[option['id'] for option in row['options']],
                    probabilities=[float(option['id'] == row['gold_option_id']) for option in row['options']])
               for row in rows]
    return rows, outputs


def test_fixture_has_independent_perturbations_and_valid_pair_rules():
    rows, outputs = fixture_predictions()
    assert len(rows) == 15
    for row in rows:
        validate_row(row)
        assert row['split'] == 'test'
        assert row['gold_option_id'] in {option['id'] for option in row['options']}
    result = evaluate(rows, outputs)
    assert result['accuracy'] == 1
    assert result['paired_semantic_consistency'] == {'n': 9, 'rate': 1}
    assert set(result['per_perturbation']) == {'base', 'reordered', 'paraphrase', 'irrelevant_context', 'missing_evidence'}


def test_consistency_excludes_changed_target_pairs():
    gold, outputs = fixture_predictions()
    for row, output in zip(gold, outputs):
        if row['perturbation']['kind'] == 'missing_evidence':
            output['probabilities'] = [1., 0., 0.]
    result = evaluate(gold, outputs)
    assert result['paired_semantic_consistency'] == {'n': 9, 'rate': 1}
    assert result['per_perturbation']['missing_evidence']['accuracy'] == 0


def test_diagnostic_joins_probabilities_by_semantic_option_id():
    gold, outputs = fixture_predictions()
    for output in outputs:
        output['option_ids'].reverse()
        output['probabilities'].reverse()
    assert evaluate(gold, outputs)['accuracy'] == 1


@pytest.mark.parametrize('bad', [[1.2, -.2, 0], [True, 0, 0], [float('nan'), .5, .5]])
def test_diagnostic_rejects_invalid_probabilities(bad):
    gold, outputs = fixture_predictions()
    outputs[0]['probabilities'] = bad
    with pytest.raises(ValueError):
        evaluate(gold, outputs)


def test_diagnostic_rejects_false_same_target_annotation():
    gold, outputs = fixture_predictions()
    gold = copy.deepcopy(gold)
    next(row for row in gold if row['perturbation']['kind'] == 'missing_evidence')['perturbation']['comparison'] = 'same_target'
    with pytest.raises(ValueError, match='semantic gold'):
        evaluate(gold, outputs)
