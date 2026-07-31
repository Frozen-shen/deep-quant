"""
Realistic transaction cost model for A-shares.

Cost components:
    - Commission: 万2.5 (0.025%) per side, min 5 yuan per trade
    - Slippage: 万5 (0.05%) market impact estimate
    - Stamp tax: 0.05% sell-side only (reduced from 0.1% on 2023-08-28)
    - Transfer fee: 0.001% both sides (沪市 only, but applied universally
      for conservative estimation)

Total round-trip cost: ~10.5bp
    Buy:  commission(2.5bp) + slippage(5bp) + transfer(0.1bp) = 7.6bp
    Sell: commission(2.5bp) + slippage(5bp) + stamp_tax(5bp) + transfer(0.1bp) = 12.6bp
    Average per side: ~10.1bp
    Round trip: ~10.5bp (weighted by typical buy/sell asymmetry)

For comparison:
    - Old system used 30bp flat (way too high, killed all alpha)
    - Institutional direct market access: 0.3-3bp (10-100x advantage)
    - Retail realistic: 10-15bp
"""


class CostModel:
    """
    Realistic transaction cost model for A-shares.

    Args:
        commission: Broker commission rate per side (default 万2.5 = 0.025%).
        slippage: Market impact / slippage estimate per side (default 万5 = 0.05%).
        stamp_tax: Stamp duty on sells only (default 0.05%, post 2023-08-28).
        transfer_fee: Transfer/registration fee both sides (default 0.001%).
        min_commission: Minimum commission per trade in yuan (default 5).
    """

    def __init__(
        self,
        commission: float = 0.00025,
        slippage: float = 0.0005,
        stamp_tax: float = 0.0005,
        transfer_fee: float = 0.00001,
        min_commission: float = 5.0,
    ):
        if any(v < 0 for v in [commission, slippage, stamp_tax, transfer_fee]):
            raise ValueError("Cost rates must be non-negative")
        if min_commission < 0:
            raise ValueError("min_commission must be non-negative")

        self.commission = commission
        self.slippage = slippage
        self.stamp_tax = stamp_tax
        self.transfer_fee = transfer_fee
        self.min_commission = min_commission

    def buy_cost(self, amount: float) -> float:
        """
        Compute total cost for a buy transaction.

        Buy costs: commission + slippage + transfer_fee.
        No stamp tax on buys.

        Args:
            amount: Trade notional value in yuan (positive).

        Returns:
            Total cost in yuan.
        """
        if amount <= 0:
            return 0.0

        commission = max(amount * self.commission, self.min_commission)
        slippage_cost = amount * self.slippage
        transfer = amount * self.transfer_fee

        return commission + slippage_cost + transfer

    def sell_cost(self, amount: float) -> float:
        """
        Compute total cost for a sell transaction.

        Sell costs: commission + slippage + stamp_tax + transfer_fee.
        Stamp tax applies only on sells.

        Args:
            amount: Trade notional value in yuan (positive).

        Returns:
            Total cost in yuan.
        """
        if amount <= 0:
            return 0.0

        commission = max(amount * self.commission, self.min_commission)
        slippage_cost = amount * self.slippage
        stamp = amount * self.stamp_tax
        transfer = amount * self.transfer_fee

        return commission + slippage_cost + stamp + transfer

    def round_trip_cost(self, amount: float) -> float:
        """
        Compute total cost for a complete round trip (buy then sell).

        Args:
            amount: Trade notional value in yuan (positive).

        Returns:
            Total round-trip cost in yuan.
        """
        return self.buy_cost(amount) + self.sell_cost(amount)

    def cost_rate(self) -> float:
        """
        Effective cost rate per trade (average of buy and sell sides).

        This is the key number for strategy viability analysis.
        At ~10.5bp per trade, a strategy needs >10.5bp alpha per
        position change to be profitable after costs.

        The average per side is used because each unit of turnover
        involves roughly half buys and half sells.

        Returns:
            Average per-side cost as a decimal fraction (e.g., 0.00105 = 10.5bp).
        """
        # For large trades (min commission is negligible)
        # Buy: commission + slippage + transfer
        # Sell: commission + slippage + stamp_tax + transfer
        buy_rate = self.commission + self.slippage + self.transfer_fee
        sell_rate = self.commission + self.slippage + self.stamp_tax + self.transfer_fee
        return (buy_rate + sell_rate) / 2.0

    def round_trip_rate(self) -> float:
        """
        Full round-trip cost rate (buy + sell combined).

        Returns:
            Total round-trip cost as a decimal fraction (~20.2bp with defaults).
        """
        buy_rate = self.commission + self.slippage + self.transfer_fee
        sell_rate = self.commission + self.slippage + self.stamp_tax + self.transfer_fee
        return buy_rate + sell_rate

    def buy_cost_rate(self) -> float:
        """Effective cost rate for a buy transaction."""
        return self.commission + self.slippage + self.transfer_fee

    def sell_cost_rate(self) -> float:
        """Effective cost rate for a sell transaction."""
        return self.commission + self.slippage + self.stamp_tax + self.transfer_fee

    def annual_cost_drag(self, annual_turnover: float) -> float:
        """
        Estimate annual cost drag given a turnover level.

        This is crucial for strategy design:
            - 300% annual turnover * 10.5bp = 31.5bp/year cost drag
            - 1000% annual turnover * 10.5bp = 105bp/year cost drag (old system)
            - Equal weight (0% turnover) = 0 cost drag (IR=0.377 baseline)

        Args:
            annual_turnover: One-way annual turnover (e.g., 3.0 = 300%).

        Returns:
            Annual cost drag as a fraction (e.g., 0.00315 = 31.5bp).
        """
        # Each unit of one-way turnover costs approximately cost_rate()
        # (the average per-side cost)
        return annual_turnover * self.cost_rate()

    def __repr__(self) -> str:
        return (
            f"CostModel(commission={self.commission:.5f}, "
            f"slippage={self.slippage:.5f}, "
            f"stamp_tax={self.stamp_tax:.5f}, "
            f"transfer_fee={self.transfer_fee:.5f}, "
            f"per_trade={self.cost_rate()*10000:.1f}bp, "
            f"round_trip={self.round_trip_rate()*10000:.1f}bp)"
        )
