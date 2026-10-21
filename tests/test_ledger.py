from profit_engine.ledger import PaperLedger


def test_ledger_conservation(tmp_path):
    db = tmp_path / "t.sqlite"
    led = PaperLedger(db)
    led.start_run("r1", "gem", {"x": 1}, initial_cash=10_000)
    led.apply_fill("r1", side="BUY", asset="SPY", shares=10, price=100, cost=5, source="test")
    pos = led.get_position("r1")
    assert pos["asset"] == "SPY"
    assert abs(pos["cash"] - (10_000 - 1000 - 5)) < 1e-9
    eq = led.mark("r1", mark_price=110, source="test")
    assert abs(eq - (pos["cash"] + 10 * 110)) < 1e-9