import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_phase3_pairs import evaluate, sha256
from placement_bandit import PlacementBandit


@pytest.fixture
def trial_data(tmp_path):
    path = tmp_path / 'memory.json'
    bandit = PlacementBandit(path)
    bandit.update({'id': 'old', 'action': 'center', 'observed_xy_m': [0.2, 0.0], 'reward': 1, 'seed': 10})
    checkpoint = 'a' * 64
    digest = sha256(path)
    manifest = {'schema': 'phase3-paired-evaluation-v1', 'heldout_seeds': [11, 12, 13],
                'checkpoint_sha256': checkpoint, 'horizon_s': 25, 'trials': []}
    for seed, success in [(11, (False, True)), (12, (True, False)), (13, (False, True))]:
        for condition, won in zip(('phase2', 'phase3_frozen'), success):
            manifest['trials'].append({'condition': condition, 'seed': seed, 'scenario': 'cube',
                'checkpoint_sha256': checkpoint, 'horizon_s': 25, 'start_xy': [0.2, .01],
                'task_success': won, 'interrupted': False, 'frozen': condition == 'phase3_frozen',
                'memory_sha256_before': digest, 'memory_sha256_after': digest})
    return manifest, path


def test_paired_counts_and_exact_probability(trial_data):
    manifest, path = trial_data
    result = evaluate(manifest, path)
    assert (result['only_phase3'], result['only_phase2'], result['both_fail']) == (2, 1, 0)
    assert result['success_difference'] == pytest.approx(1/3)
    assert result['mcnemar_exact_two_sided_p'] == 1.0


def test_heldout_seed_overlap_rejected(trial_data):
    manifest, path = trial_data
    manifest['heldout_seeds'][0] = 10
    with pytest.raises(ValueError, match='overlap'):
        evaluate(manifest, path)


@pytest.mark.parametrize('mutation, message', [
    (lambda t: t.update(memory_sha256_after='b'*64), 'frozen'),
    (lambda t: t.update(frozen=False), 'frozen'),
    (lambda t: t.update(interrupted=True), 'interrupted'),
    (lambda t: t.update(horizon_s=21), 'horizon'),
    (lambda t: t.update(start_xy=[0.3, 0.1]), 'starting positions'),
    (lambda t: t.update(scenario='sphere'), 'scenario'),
])
def test_mismatches_rejected(trial_data, mutation, message):
    manifest, path = trial_data
    mutation(manifest['trials'][1])
    with pytest.raises(ValueError, match=message):
        evaluate(manifest, path)


def test_missing_and_duplicate_runs_rejected(trial_data):
    manifest, path = trial_data
    missing = copy.deepcopy(manifest)
    missing['trials'].pop()
    with pytest.raises(ValueError, match='missing'):
        evaluate(missing, path)
    manifest['trials'].append(copy.deepcopy(manifest['trials'][0]))
    with pytest.raises(ValueError, match='duplicate'):
        evaluate(manifest, path)


def test_bandit_matches_sherman_morrison_single_observation(tmp_path):
    # Closed-form rank-one ridge update, independently derived from (I+zzᵀ)^-1.
    from placement_bandit import ALPHA
    import math
    bandit = PlacementBandit(tmp_path / 'memory.json')
    observation = [.22, .09]
    query = [.18, -.09]
    bandit.update({'id': 'a', 'action': 'center', 'observed_xy_m': observation, 'reward': 1, 'seed': 1})
    z = [1., .2, .5]
    x = [1., -.2, -.5]
    dot = sum(a * b for a, b in zip(x, z))
    zz = sum(a*a for a in z)
    xx = sum(a*a for a in x)
    mean = dot / (1 + zz)
    uncertainty = math.sqrt(xx - dot*dot / (1 + zz))
    score = bandit.scores(query)['center']
    assert score['mean'] == pytest.approx(mean)
    assert score['uncertainty'] == pytest.approx(uncertainty)
    assert score['score'] == pytest.approx(mean + ALPHA*uncertainty)
    assert bandit.scores(query)['x_plus']['mean'] == 0
    assert bandit.scores(query)['x_plus']['uncertainty'] == pytest.approx(math.sqrt(xx))


def test_untraceable_memory_seed_rejected(tmp_path, trial_data):
    manifest, _ = trial_data
    path = tmp_path / 'untraceable.json'
    PlacementBandit(path).update({'id': 'x', 'action': 'center', 'observed_xy_m': [0.2, 0.0], 'reward': 1})
    with pytest.raises(ValueError, match='unknown training seeds'):
        evaluate(manifest, path)


def test_exact_p_value_for_all_discordant_pairs_in_same_direction(trial_data):
    manifest, path = trial_data
    for row in manifest['trials']:
        row['task_success'] = row['condition'] == 'phase3_frozen'
    result = evaluate(manifest, path)
    assert result['success_difference'] == 1.0
    assert result['mcnemar_exact_two_sided_p'] == pytest.approx(0.25)
