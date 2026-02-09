"""
Phased Mean-Reversion Accumulator Strategy
Author: DeepSeek Labs, Inc.
Description: Symmetrical accumulation strategy for both base and quote assets
"""
# from typing import Dict, List, Optional, Tuple
from typing import Dict, List
from decimal import Decimal

# import pandas as pd
import numpy as np

from pydantic import BaseModel

# from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.common import OrderType, PriceType
# from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.connector.connector_base import ConnectorBase


class PhasedMeanReversionAccumulator(ScriptStrategyBase):
    """
    Implements the Phased Mean-Reversion Accumulator algorithm.

    The strategy simultaneously accumulates both base and quote assets
    through symmetrical, independent accumulation cycles.
    """

    # Configuration parameters
    exchange = "binance"  # Change to your exchange
    trading_pair = "BTC-USDT"  # Change to your pair
    fields : List[str] = trading_pair.split("-")
    base_asset = fields[0]  # BA
    quote_asset = fields[1]  # QA

    # Accumulation settings
    ba_amount_to_sell = Decimal("0.001")  # Amount of BA to sell in transformation phase
    qa_amount_to_spend = Decimal("100")   # Amount of QA to spend in transformation phase
    price_delta = Decimal("0.001")        # Delta for price adjustment (0.1%)
    advantage_factor = Decimal("0.002")   # Advantage factor for calculations (0.2%)

    # Rolling average settings
    price_history_window = 100  # Number of ticks to average
    sampling_interval = 5      # Ticks between sampling

    @classmethod
    def init_markets(cls, config: BaseModel):
        pass

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)

        # State variables for BA accumulation
        self.ba_state = "idle"  # idle, transformation, calculation, going_back
        self.ba_avg_bid = None
        self.ba_sell_price = None
        self.ba_buy_price = None
        self.ba_sell_order_id = None
        self.ba_buy_order_id = None
        self.ba_accumulated_amount = Decimal("0")

        # State variables for QA accumulation
        self.qa_state = "idle"  # idle, transformation, calculation, going_back
        self.qa_avg_ask = None
        self.qa_buy_price = None
        self.qa_sell_price = None
        self.qa_buy_order_id = None
        self.qa_sell_order_id = None
        self.qa_accumulated_amount = Decimal("0")

        # Price history for rolling averages
        self.bid_history : List[float] = []
        self.ask_history : List[float]= []
        self.sample_counter = 0

        # Tracking
        self.ba_cycles_completed = 0
        self.qa_cycles_completed = 0

    def on_tick(self):
        """Main strategy execution loop"""
        # Step 1: Update rolling averages
        self._update_price_averages()

        # Execute BA accumulation cycle
        self._execute_ba_accumulation()

        # Execute QA accumulation cycle
        self._execute_qa_accumulation()

        # Log status periodically
        if self.sample_counter % 20 == 0:
            self.logger().info(self.format_status())

    def _update_price_averages(self):
        """Step 1: Calculate rolling averages of best bid/ask"""
        self.sample_counter += 1

        if self.sample_counter % self.sampling_interval == 0:
            # Get current market prices
            bid_price = self.connectors[self.exchange].get_price_by_type(
                self.trading_pair, PriceType.BestBid
            )
            ask_price = self.connectors[self.exchange].get_price_by_type(
                self.trading_pair, PriceType.BestAsk
            )

            if bid_price and ask_price:
                self.bid_history.append(float(bid_price))
                self.ask_history.append(float(ask_price))

                # Maintain rolling window
                if len(self.bid_history) > self.price_history_window:
                    self.bid_history.pop(0)
                    self.ask_history.pop(0)

                # Calculate averages
                if len(self.bid_history) > 0:
                    self.ba_avg_bid = Decimal(str(np.mean(self.bid_history)))
                    self.qa_avg_ask = Decimal(str(np.mean(self.ask_history)))

    def _execute_ba_accumulation(self):
        """Execute BA (Base Asset) accumulation cycle"""

        # Transformation Phase (Step 2)
        if self.ba_state == "idle" and self.ba_avg_bid:
            # Check if we have enough BA to sell
            ba_balance = self.connectors[self.exchange].get_balance(self.base_asset)

            if ba_balance >= self.ba_amount_to_sell:
                # Calculate sell price: avg_bid - delta
                self.ba_sell_price = self.ba_avg_bid * (Decimal("1") - self.price_delta)

                # Place sell order
                self.ba_sell_order_id = self.sell(
                    connector_name=self.exchange,
                    trading_pair=self.trading_pair,
                    amount=self.ba_amount_to_sell,
                    order_type=OrderType.LIMIT,
                    price=self.ba_sell_price
                )

                if self.ba_sell_order_id:
                    self.ba_state = "transformation"
                    self.logger().info("BA: Started transformation phase - Selling %f at %f",
                                       self.ba_amount_to_sell, self.ba_sell_price)

        # Calculation Phase (Step 3) - triggered after sell order fills
        elif self.ba_state == "calculation":
            # Calculate new buy price: recover 2 operations + advantage
            # Formula: buy_price = sell_price * (1 - 2*fee - advantage)
            # Assuming 0.1% fee per operation
            fee_per_op = Decimal("0.001")
            total_cost_factor = Decimal("1") - Decimal("2") * fee_per_op - self.advantage_factor

            self.ba_buy_price = self.ba_sell_price * total_cost_factor

            self.ba_state = "going_back"
            self.logger().info("BA: Calculation complete - Will buy back at %f", self.ba_buy_price)

        # Going Back Phase (Step 4)
        elif self.ba_state == "going_back" and self.ba_buy_price:
            # Place buy order to accumulate BA
            qa_balance = self.connectors[self.exchange].get_balance(self.quote_asset)
            max_buy_amount = qa_balance / self.ba_buy_price

            if max_buy_amount >= self.ba_amount_to_sell:  # Buy back what we sold plus more
                buy_amount = self.ba_amount_to_sell * (Decimal("1") + self.advantage_factor)

                self.ba_buy_order_id = self.buy(
                    connector_name=self.exchange,
                    trading_pair=self.trading_pair,
                    amount=buy_amount,
                    order_type=OrderType.LIMIT,
                    price=self.ba_buy_price
                )

                if self.ba_buy_order_id:
                    self.logger().info("BA: Going back phase - Buying %f at %f",
                                       buy_amount, self.ba_buy_price)

    def _execute_qa_accumulation(self):
        """Execute QA (Quote Asset) accumulation cycle"""

        # Transformation Phase (Step 2)
        if self.qa_state == "idle" and self.qa_avg_ask:
            # Check if we have enough QA to spend
            qa_balance = self.connectors[self.exchange].get_balance(self.quote_asset)

            if qa_balance >= self.qa_amount_to_spend:
                # Calculate buy price: avg_ask + delta
                self.qa_buy_price = self.qa_avg_ask * (Decimal("1") + self.price_delta)

                # Calculate amount of BA to buy
                ba_amount_to_buy = self.qa_amount_to_spend / self.qa_buy_price

                # Place buy order
                self.qa_buy_order_id = self.buy(
                    connector_name=self.exchange,
                    trading_pair=self.trading_pair,
                    amount=ba_amount_to_buy,
                    order_type=OrderType.LIMIT,
                    price=self.qa_buy_price
                )

                if self.qa_buy_order_id:
                    self.qa_state = "transformation"
                    self.logger().info("QA: Started transformation phase - Buying %f BA at %f",
                                       ba_amount_to_buy, self.qa_buy_price)

        # Calculation Phase (Step 3) - triggered after buy order fills
        elif self.qa_state == "calculation":
            # Calculate new sell price: recover 2 operations + advantage
            # Formula: sell_price = buy_price * (1 + 2*fee + advantage)
            fee_per_op = Decimal("0.001")
            total_gain_factor = Decimal("1") + Decimal("2") * fee_per_op + self.advantage_factor

            self.qa_sell_price = self.qa_buy_price * total_gain_factor

            self.qa_state = "going_back"
            self.logger().info("QA: Calculation complete - Will sell at %f", self.qa_sell_price)

        # Going Back Phase (Step 4)
        elif self.qa_state == "going_back" and self.qa_sell_price:
            # Get the amount of BA we bought
            ba_balance = self.connectors[self.exchange].get_balance(self.base_asset)

            if ba_balance > Decimal("0"):
                # Sell the BA at calculated price
                sell_amount = ba_balance  # Sell all BA from this cycle

                self.qa_sell_order_id = self.sell(
                    connector_name=self.exchange,
                    trading_pair=self.trading_pair,
                    amount=sell_amount,
                    order_type=OrderType.LIMIT,
                    price=self.qa_sell_price
                )

                if self.qa_sell_order_id:
                    self.logger().info("QA: Going back phase - Selling %f at %f",
                                       sell_amount, self.qa_sell_price)

    def did_fill_order(self, event: OrderFilledEvent):
        """Handle order fill events"""
        # BA accumulation events
        if event.order_id == self.ba_sell_order_id:
            self.logger().info("BA: Transformation sell order filled at %f", event.price)
            self.ba_state = "calculation"
            self.qa_accumulated_amount += event.amount * event.price  # Track QA received

        elif event.order_id == self.ba_buy_order_id:
            self.logger().info("BA: Going back buy order filled at %f", event.price)
            self.ba_state = "idle"
            self.ba_accumulated_amount += event.amount
            self.ba_cycles_completed += 1

        # QA accumulation events
        elif event.order_id == self.qa_buy_order_id:
            self.logger().info("QA: Transformation buy order filled at %f", event.price)
            self.qa_state = "calculation"
            self.ba_accumulated_amount += event.amount  # Track BA received

        elif event.order_id == self.qa_sell_order_id:
            self.logger().info("QA: Going back sell order filled at %f", event.price)
            self.qa_state = "idle"
            self.qa_accumulated_amount += event.amount * event.price
            self.qa_cycles_completed += 1

    def format_status(self) -> str:
        """Return formatted status for UI"""
        if not self.ready_to_trade:
            return "Market connectors are not ready."

        lines = []
        warning_lines : List[str] = []

        # Market info
        lines.append(f"Exchange: {self.exchange} | Trading Pair: {self.trading_pair}")
        lines.append("-" * 50)

        # Price info
        if self.ba_avg_bid and self.qa_avg_ask:
            lines.append(f"Current Avg Bid: {self.ba_avg_bid:.8f}")
            lines.append(f"Current Avg Ask: {self.qa_avg_ask:.8f}")
            spread : float = (self.qa_avg_ask - self.ba_avg_bid) / self.ba_avg_bid * 100 \
                if self.ba_avg_bid > 0 else 0
            lines.append(f"Spread: {spread:.2f}%")
        lines.append("")

        # BA Accumulation Status
        lines.append("=== BA (Base Asset) Accumulation ===")
        lines.append(f"State: {self.ba_state}")
        lines.append(f"Cycles Completed: {self.ba_cycles_completed}")
        lines.append(f"Accumulated BA: {self.ba_accumulated_amount:.8f}")
        if self.ba_sell_price:
            lines.append(f"Last Sell Price: {self.ba_sell_price:.8f}")
        if self.ba_buy_price:
            lines.append(f"Target Buy Price: {self.ba_buy_price:.8f}")
        lines.append("")

        # QA Accumulation Status
        lines.append("=== QA (Quote Asset) Accumulation ===")
        lines.append(f"State: {self.qa_state}")
        lines.append(f"Cycles Completed: {self.qa_cycles_completed}")
        lines.append(f"Accumulated QA: {self.qa_accumulated_amount:.2f}")
        if self.qa_buy_price:
            lines.append(f"Last Buy Price: {self.qa_buy_price:.8f}")
        if self.qa_sell_price:
            lines.append(f"Target Sell Price: {self.qa_sell_price:.8f}")
        lines.append("")

        # Balances
        ba_balance = self.connectors[self.exchange].get_balance(self.base_asset)
        qa_balance = self.connectors[self.exchange].get_balance(self.quote_asset)
        lines.append(f"Balances - {self.base_asset}: {ba_balance:.8f} | " +
                     f"{self.quote_asset}: {qa_balance:.2f}")

        # Warnings
        if len(warning_lines) > 0:
            lines.append("")
            lines.append("*** WARNINGS ***")
            for warning in warning_lines:
                lines.append(f"  {warning}")

        return "\n".join(lines)
