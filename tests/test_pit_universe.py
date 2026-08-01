"""tests/test_pit_universe.py — Tests for PIT Universe Builder."""

from data.pit_universe import get_universe, get_all_trading_stocks


class TestGetUniverse:
    def test_returns_list_of_codes(self):
        universe = get_universe("2020-01-15")
        assert isinstance(universe, list)
        assert len(universe) > 500
        assert all(isinstance(c, str) and len(c) == 6 for c in universe)

    def test_different_dates_different_universes(self):
        # Date within constituent data range vs date before data starts (fallback)
        u1 = get_universe("2020-01-15")
        u2 = get_universe("2010-01-01")
        assert set(u1) != set(u2)

    def test_fallback_for_old_date(self):
        # Date before constituent data starts
        universe = get_universe("2010-01-01")
        assert len(universe) > 0


class TestGetAllTradingStocks:
    def test_returns_codes(self):
        stocks = get_all_trading_stocks()
        assert len(stocks) > 1000
        assert all(len(c) == 6 for c in stocks)
