import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from hold_verifier import verify_hold


def test_fusion_requires_agreeing_cues_and_abstains_on_conflict():
    red = np.zeros((240,320,3), dtype=np.uint8)
    red[100:160,120:180] = (255, 0, 0)
    blank = np.zeros_like(red)
    assert verify_hold([red]*3, [.15]*3) == 'held'
    assert verify_hold([blank]*3, [-.1]*3) == 'empty'
    assert verify_hold([blank]*3, [.15]*3) == 'unknown'
    assert verify_hold([red]*3, [-.1]*3) == 'unknown'
    assert verify_hold([red, blank, red], [.15]*3) == 'unknown'


def test_gripper_vision_ablation_and_invalid_inputs():
    blank = np.zeros((240,320,3), dtype=np.uint8)
    assert verify_hold([blank]*3, [.15]*3, mode='gripper') == 'held'
    assert verify_hold([blank]*3, [-.1]*3, mode='vision') == 'unknown'
    assert verify_hold([blank]*2, [.15]*2) == 'unknown'
    assert verify_hold([blank]*3, [.15,float('nan'),.15]) == 'unknown'


def test_truth_label_keeps_ambiguous_cases_out_of_accuracy():
    from evaluate_hold_verifier import truth_label
    assert truth_label([.09]*5, [True]*5, .015) == 'held'
    assert truth_label([.015]*5, [False]*5, .015) == 'empty'
    assert truth_label([.09]*5, [False]*5, .015) == 'unknown'
    assert truth_label([.015]*5, [True]*5, .015) == 'unknown'


def test_delay_is_signed_and_unknown_when_no_loss_reference():
    from evaluate_hold_verifier import loss_detection
    red = np.zeros((240,320,3), dtype=np.uint8)
    red[100:160,120:180] = (255,0,0)
    blank = np.zeros_like(red)
    time = np.arange(12, dtype=float) * .05
    z = np.array([.1]*6 + [.015]*6)
    contacts = np.array([True]*6 + [False]*5)
    wrist = np.array([red]*5 + [blank]*7)
    positions = np.array([.15]*5 + [-.1]*7)
    metrics = loss_detection(time, z, contacts, wrist, positions, .015, 5, 11)
    assert metrics['reference_empty_time_s'] is not None
    assert metrics['fusion_empty_time_s'] is not None
    assert metrics['fusion_signed_delay_s'] == metrics['fusion_empty_time_s'] - metrics['reference_empty_time_s']
    no_loss = loss_detection(time, np.full(12,.1), np.ones(11, dtype=bool), wrist, positions, .015, 5, 11)
    assert no_loss['reference_empty_time_s'] is None
    assert no_loss['fusion_signed_delay_s'] is None


def test_closure_truth_does_not_require_lift():
    from evaluate_hold_verifier import truth_label
    assert truth_label([.015]*5, [True]*5, .015, 'after_closure') == 'held'
    assert truth_label([.015]*5, [True]*5, .015, 'after_lift') == 'unknown'
