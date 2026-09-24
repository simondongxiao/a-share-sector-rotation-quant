from __future__ import annotations

import pandas as pd

from dynamic_exit_engine import DynamicExitConfig, evaluate_dynamic_exit


def make_frame(rows):
    return pd.DataFrame(rows).set_index("date")


def test_panic_clear():
    data = make_frame([
        {"date": "2026-09-21", "Position": 1.0, "State": "OVERHEATED", "Limit_Down_Count": 0, "Close": 10, "Low": 9.8, "High": 10.2, "Daily_Drop": 0.01, "MA5": 9.9, "MA10": 9.7},
        {"date": "2026-09-22", "Position": 1.0, "State": "OVERHEATED", "Limit_Down_Count": 3, "Close": 9.4, "Low": 9.35, "High": 10.0, "Daily_Drop": -0.06, "MA5": 9.8, "MA10": 9.7},
    ])
    decision = evaluate_dynamic_exit(data, pd.DataFrame(index=data.index), "2026-09-22")
    assert decision.new_position == 0.0
    assert decision.action == "PANIC_CLEAR"


def test_partial_reduction_with_shadow():
    data = make_frame([
        {"date": "2026-09-21", "Position": 1.0, "State": "OVERHEATED", "Limit_Down_Count": 0, "Close": 10, "Low": 9.8, "High": 10.2, "Daily_Drop": 0.01, "MA5": 9.9, "MA10": 9.7},
        {"date": "2026-09-22", "Position": 1.0, "State": "OVERHEATED", "Limit_Down_Count": 0, "Close": 9.7, "Low": 9.0, "High": 10.0, "Daily_Drop": -0.04, "MA5": 9.8, "MA10": 9.7},
    ])
    decision = evaluate_dynamic_exit(data, pd.DataFrame(index=data.index), "2026-09-22")
    assert decision.new_position == 0.5
    assert decision.action == "PARTIAL_REDUCTION"


def test_rebound_after_reduction():
    data = make_frame([
        {"date": "2026-09-21", "Position": 0.5, "State": "OVERHEATED", "Limit_Down_Count": 0, "Close": 9.4, "Low": 9.2, "High": 9.6, "Daily_Drop": -0.04, "MA5": 9.8, "MA10": 9.7},
        {"date": "2026-09-22", "Position": 0.5, "State": "NORMAL", "Limit_Down_Count": 0, "Close": 10.0, "Low": 9.7, "High": 10.1, "Daily_Drop": 0.03, "MA5": 9.8, "MA10": 9.7},
    ])
    decision = evaluate_dynamic_exit(data, pd.DataFrame(index=data.index), "2026-09-22")
    assert decision.new_position == 1.0
    assert decision.action == "REBOUND_REENTRY"


def test_a_kill_confirmation():
    data = make_frame([
        {"date": "2026-09-21", "Position": 0.5, "State": "OVERHEATED", "Limit_Down_Count": 0, "Close": 9.4, "Low": 9.2, "High": 9.6, "Daily_Drop": -0.04, "MA5": 9.8, "MA10": 9.7},
        {"date": "2026-09-22", "Position": 0.5, "State": "NORMAL", "Limit_Down_Count": 0, "Close": 9.5, "Low": 9.4, "High": 9.6, "Daily_Drop": -0.01, "MA5": 9.7, "MA10": 9.7},
    ])
    decision = evaluate_dynamic_exit(data, pd.DataFrame(index=data.index), "2026-09-22")
    assert decision.new_position == 0.0
    assert decision.action == "CONFIRM_A_KILL"


def test_missing_limit_down_count_does_not_claim_panic_clear():
    data = make_frame([
        {"date": "2026-09-21", "Position": 1.0, "State": "OVERHEATED", "Limit_Down_Count": None, "Close": 10, "Low": 9.8, "High": 10.2, "Daily_Drop": 0.01, "MA5": 9.9, "MA10": 9.7},
        {"date": "2026-09-22", "Position": 1.0, "State": "OVERHEATED", "Limit_Down_Count": None, "Close": 9.4, "Low": 9.35, "High": 10.0, "Daily_Drop": -0.06, "MA5": 9.8, "MA10": 9.7},
    ])
    decision = evaluate_dynamic_exit(data, pd.DataFrame(index=data.index), "2026-09-22")
    assert decision.new_position == 0.2
    assert decision.action == "ORDINARY_REDUCTION"
    assert decision.data_quality == "missing_limit_down_count"


if __name__ == "__main__":
    test_panic_clear()
    test_partial_reduction_with_shadow()
    test_rebound_after_reduction()
    test_a_kill_confirmation()
    test_missing_limit_down_count_does_not_claim_panic_clear()
    print("dynamic exit engine tests passed")
