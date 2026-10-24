from parali.app.explain import describe


def test_describe_fills_values():
    assert describe("ring1_3d", 12) == "12 fire(s) in the 6 surrounding cells over 3 days"


def test_describe_handles_today_and_missing_values():
    assert describe("days_since_fire", 0) == "A fire was detected in this cell today"
    assert describe("history_rate", None) == "history rate"
    assert describe("unknown_feature", 1) == "unknown feature"
