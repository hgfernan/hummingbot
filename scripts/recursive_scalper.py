#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: recursive_scalper.py
Created: 2026-03030 17:22:58
@author: @hgfernan
Description: Implement the Round Trip Trading algorithm
"""

# from datetime import datetime
# import enum  # class Enum, auto()
# import logging  # class Logger, getLogger()

# import json
import os

# from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel

from hummingbot.client.hummingbot_application import HummingbotApplication

# from hummingbot import data_path
# from hummingbot.connector.connector_base import ConnectorBase
# from hummingbot.core.event.event_forwarder import SourceInfoEventForwarder
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    MarketOrderFailureEvent,
    OrderCancelledEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    SellOrderCompletedEvent,
)
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from scripts.utility.recursive_scalper_helper import (  # ActivityParams,; PriceAmount,
    Accumulator,
    OrderParams,
    QuoteAccumulator,
    RestoreParams,
    RsState,
    TransformParams,
)

# class QuoteAccumulator(Accumulator):
#     """
#     Accumulator of the quote asset
#     """

#     last_instance_id: int = 0

#     def __init__(self,
#                  starting_tick: int,
#                  transform_params: TransformParams,
#                  restore_params: RestoreParams) -> None:
#         """
#         Initialize a QuoteAccumulator instance
#         """

#         super().__init__(starting_tick,
#                          transform_params=transform_params,
#                          restore_params=restore_params)

#         self.id: int = self.__class__._inc_instance_id()

#         self.local_logger: Optional[logging.Logger] = None

#     def logger(self) -> logging.Logger:
#         """
#         Return this class logger. Create one if necessary.
#         """
#         if self.local_logger is None:
#             self.local_logger = logging.getLogger(self.__class__.__name__)

#         # Normal function termination
#         return self.local_logger

#     def calc_transform_price(self) -> Decimal:
#         """
#         Calculate the transformation price, using base_price and rel_delta
#         """
#         result: Decimal = self.get_base_price() * (1 - self.transform_params.rel_delta)

#         self.curr_price_amount.price = result

#         # HINT Normal function termination
#         return result

#     def calc_restore_price(self) -> Decimal:
#         """
#         Calculate the restoration price, using curr_price, exchange_fee and
#         gain_ratio
#         """

#         exchange_fee: Decimal = self.restore_params.exchange_fee
#         gain_ratio: Decimal = self.restore_params.gain_ratio

#         denom: float = 1.0 - float(exchange_fee)
#         denom *= denom

#         result: Decimal = Decimal(1.0) + gain_ratio * exchange_fee
#         result /= Decimal(denom)

#         # HINT Normal function termination
#         return result

#     def execute_accumulation(self) -> OrderParams:
#         """
#         Manage the finite state automaton for the accumulation, and keep
#         a log of gain and loss
#         """

#         result: OrderParams = OrderParams()

#         state: RsState = self.get_current_state()
#         match state:
#             case RsState.START:
#                 self.logger().info(state.name)

#                 result.state_name = state.get_description()

#                 # TODO what else should be done here ?

#                 self.set_current_state(RsState.TRANSFORM_ACTION)

#             case RsState.TRANSFORM_CALC:
#                 self.logger().info(state.name)

#                 result.state_name = state.get_description()

#                 # TODO calculate price and format order

#                 self.set_current_state(RsState.TRANSFORM_ACTION)

#             case RsState.TRANSFORM_ACTION:
#                 self.logger().info(state.name)

#                 result.side = "buy"
#                 result.state_name = state.get_description()

#                 # TODO wait here till buy order is complete
#                 # HINT the state change will be caused by did_complete_buy_order()

#             case RsState.RESTORE_CALC:
#                 self.logger().info(state.name)

#                 result.state_name = state.get_description()

#             case RsState.RESTORE_ACTION:
#                 self.logger().info(state.name)

#                 result.side = "sell"
#                 result.state_name = state.get_description()

#             case RsState.STOP:
#                 self.logger().info(state.name)

#                 result.state_name = state.get_description()

#             case _:
#                 msg: str = ''

#                 if isinstance(state, RsState):
#                     msg = f"Invalid or unknown state: {state.name} ({state.value})"

#                 else:
#                     msg = f"Invalid or unknown state: {state}"

#                 self.logger().error(msg)

#                 result.state_name = msg

#         # HINT Normal function termination
#         return result

#     # def did_create_buy_order(self, event: BuyOrderCreatedEvent):
#         # """
#         # Future versions will confirm a buy order was really issued
#         # """
#         #
#         # pass

#     # def did_create_sell_order(self, event: SellOrderCreatedEvent):
#         # """
#         # Future versions will confirm a sell order was really issued
#         # """
#         #
#         # pass

#     def did_fill_order(self, event: OrderFilledEvent):
#         """
#         Register partial order fills, till the order fullfilling.

#         Won't issue a complete order event.
#         """

#         self.partial_fills.append(PriceAmount(price=event.price, amount=event.amount))

#     def did_fail_order(self, event: MarketOrderFailureEvent):
#         """
#         This helper will go to the STOP state if it receives this message, and
#         will stop further processing, and won't be seen by the master anymore.
#         """

#     def did_cancel_order(self, event: OrderCancelledEvent):
#         """
#         This helper will go to the STOP state if it receives this message, and
#         will stop further processing, and won't be seen by the master anymore.
#         """

#     def did_expire_order(self, event: OrderExpiredEvent):
#         """
#         This helper will go to the STOP state if it receives this message, and
#         will stop further processing, and won't be seen by the master anymore.
#         """

#     def did_complete_buy_order(self, event: BuyOrderCompletedEvent):
#         """
#         A `QuoteAccumulator` in `RsState.TRANSFORM_ACTION` will go to
#         `RsState.RESTORE_CALC`; a `BaseAccumulator` in `RsState.RESTORE_ACTION`
#         will go to `RsState.TRANSFORM_CALC`
#         """

#         self.set_current_state(RsState.RESTORE_CALC)

#     def did_complete_sell_order(self, event: SellOrderCompletedEvent):
#         """
#         A `QuoteAccumulator` in `RsState.RESTORE_ACTION` will go to
#         `RsState.TRANSFORM_CALC`; a `BaseAccumulator` in `RsState.TRANSFORM_ACTION`
#         will go to `RsState.RESTORE_CALC`
#         """

#         self.set_current_state(RsState.RESTORE_CALC)


class RecursiveAccumulator(ScriptStrategyBase):
    """
    Manage the creation of accumulators, according to configuration, and their
    execution
    """

    # exchange: str = str(os.getenv("EXCHANGE", "binance_paper_trade"))
    exchange: str = os.getenv("EXCHANGE", "binance_paper_trade")
    trading_pair: str = str(os.getenv("TRADING_PAIRS", "ETH-USDT"))

    depth: Any = os.getenv("DEPTH", "50")
    if isinstance(depth, str) and depth.isdigit():
        depth = int(depth)
    else:
        raise ValueError("DEPTH must be an integer")

    # last_dump_timestamp = 0
    # time_between_csv_dumps = 10

    # obook_temp_storage = trading_pair
    # trades_temp_storage = trading_pair
    # current_date = None
    # market = {exchange: trading_pair}
    # subscribed_to_order_book_trade_event: bool = False

    max_ticks: int = 10
    should_stop: bool = False

    # HINT mean price of base asset where other price estimaates will be derived from
    mean_base_price: Decimal = Decimal(0.0)

    quote_investment: Decimal = Decimal(25.0)
    quote_ratio: Decimal = Decimal(0.05)

    base_investment: Decimal = Decimal(25.0 / 2000)
    base_ratio: Decimal = Decimal(0.05)

    # HINT exchange fee
    fee: Decimal = Decimal(0.001)

    # HINT gain ratio over exchange fee
    gain: Decimal = Decimal(0.5)

    # HINT number of QuoteAccumulator helpers
    n_quote_accumulators: int = 1

    # HINT number of BaseAccumulator helpers
    n_base_accumulators: int = 1

    # HINT ticks between the launch of each helper
    n_ticks_between: int = 30

    @classmethod
    def init_markets(cls, config: BaseModel):
        """
        This method is called in the start command if the script has a config class defined,
        and allows script to define the market connectors and trading pairs needed for the
        strategy operation.
        """
        cls.markets = {cls.exchange: set(cls.trading_pairs)}

    def get_base_name(self) -> str:
        """
        Get the name of the base asset
        """

        return self.base_name

    def get_quote_name(self) -> str:
        """
        Get the name of the quote asset
        """

        return self.quote_name

    def get_mean_base_price(self) -> Decimal:
        """
        Return the suggested mean base price.
        """

        return self.mean_base_price

    def __init__(self, connectors: Dict[str, Any], config: Optional[BaseModel] = None):
        """
        Initialization of the RecursiveAccumulator instance.

        In future versions, parameters will be received from a configuration object.
        """

        super().__init__(connectors, config)

        fields: List[str] = self.trading_pair.split("-")

        self.base_name: str = fields[0]
        self.quote_name: str = fields[1]

        self.avail_balances: Dict[str, Decimal] = {}

        rv: bool = self.retrieve_balances()
        if not rv:
            self.logger().critical("Could not find balance information for all assets")

            HummingbotApplication.main_application().stop()

        expected_balance: Decimal = self.base_investment * self.n_base_accumulators
        if expected_balance < self.avail_balances[self.base_name]:
            self.logger().critical("Not enough balance for %d base accumulators. " +
                                   "%f would be necessary",
                                   self.n_base_accumulators, expected_balance)

            HummingbotApplication.main_application().stop()

        expected_balance = self.quote_investment * self.n_quote_accumulators
        if expected_balance < self.avail_balances[self.quote_name]:
            self.logger().critical("Not enough balance for %d quote accumulators. " +
                                   "%f would be necessary",
                                   self.n_quote_accumulators, expected_balance)

            HummingbotApplication.main_application().stop()

        # HINT set of active Accumulator objects
        self.active_helpers: Set[Accumulator] = set()

        # HINT dictionary of active orders for Accumulator objects
        self.active_orders: Dict[str, Accumulator] = {}

        self.tick_counter: int = -1

        self.estimate_params()

        # HINT set up quote accumulators
        transform_params: TransformParams = \
            TransformParams(investment=self.quote_investment,
                            base_price=self.mean_base_price,
                            rel_delta=self.rel_delta)

        restore_params: RestoreParams = \
            RestoreParams(exchange_fee=self.fee, gain_ratio=self.gain_ratio)

        # HINT cumulative tickers
        cum_ticks: int = 0
        n_common_accumulators: int = min(self.n_base_accumulators, self.n_quote_accumulators)
        for _ in range(n_common_accumulators):
            qa: QuoteAccumulator = QuoteAccumulator(cum_ticks, transform_params, restore_params)

            if not self.add_helper(qa):
                self.logger().error("FATAL Could not add QuoteAccumulator %d", qa.instance_name())

                HummingbotApplication.main_application().stop()

            cum_ticks += self.n_ticks_between

            # ba: BaseAccumulator = BaseAccumulator(cum_ticks, transform_params, restore_params)

            # if not self.add_helper(ba):
            #     self.logger().error("FATAL Could not add QuoteAccumulator %d", ba.instance_name())

            #     HummingbotApplication.main_application().stop()

            # cum_ticks += self.n_ticks_between
            # self.starting_tick.append(cum_ticks)

        for __ind in range(self.n_quote_accumulators - n_common_accumulators):
            qr: QuoteAccumulator = QuoteAccumulator(cum_ticks, transform_params, restore_params)

            if not self.add_helper(qr):
                self.logger().error("FATAL Could not add QuoteAccumulator %d", qr.instance_name())

                HummingbotApplication.main_application().stop()

            cum_ticks += self.n_ticks_between
            self.starting_tick.append(cum_ticks)

        # HINT set up base accumulators
        # transform_params = \
        #     TransformParams(investment=self.base_investment,
        #                     base_price=self.mean_base_price,
        #                     rel_delta=self.rel_delta)

        # for ba_ind in range(self.n_quote_accumulators - n_common_accumulators):
        #     ba: BaseAccumulator = BaseAccumulator(cum_ticks, transform_params, restore_params)

        #     if not self.add_helper(ba):
        #         self.logger().error("FATAL Could not add QuoteAccumulator %d", ba.instance_name())

        #         HummingbotApplication.main_application().stop()

        #     cum_ticks += self.n_ticks_between
        #     self.starting_tick.append(cum_ticks)

    def place_order(self, params: OrderParams) -> str:
        """
        Select `RecursiveAccumulator` methods `buy()` and `sell()` according to the order side
        """

        result: str = ""
        side: str = params.side.lower()
        if side not in ["buy", "sell"]:
            return ""

        params_dict = params.model_dump()
        params_dict.pop("state")
        params_dict.pop("side")

        if "buy" == params.side.lower():
            result = self.buy(**params_dict)

        elif "sell" == params.side.lower():
            result = self.sell(**params_dict)

        # HINT Normal function termination
        return result

    def _set_balance(self, asset_name: str) -> bool:
        """
        Set the balance of the given asset name, and return True if successful
        """

        aux: Any = self.connectors[self.exchange].get_balance(asset_name)

        if not isinstance(aux, float):
            self.logger().error("Invalid or unkwon asset name %s", asset_name)

            self.avail_balances[asset_name] = Decimal(0.0)

            # HINT return to indicate failure
            return False

        self.avail_balances[asset_name] = Decimal(float(aux))

        # HINT Normal function termination
        return True

    def retrieve_balances(self) -> bool:
        """
        Obtain the availab
        """

        result: bool = True

        result = result and self._set_balance(self.base_name)
        if not result:
            self.logger().error("Could not find the balance of base asset %s", self.base_name)

        result = result and self._set_balance(self.quote_name)
        if not result:
            self.logger().error("Could not find the balance of quote asset %s", self.quote_name)

        # HINT Normal function termination
        return result

    def estimate_params(self) -> None:
        """
        Estimate statistical parameters -- mostly the base price and the relative
        delta to adjust the base price
        """

        # TODO calculate base price from order book and trade info
        self.mean_base_price = Decimal(2000.0)

    def on_tick(self) -> None:
        """
        Main program of the strategy: will receive Hummingbot ticks till there are
        no more helpers to process, or while there are still iterations to run.
        """

        order_params: OrderParams = OrderParams()

        # TODO add orderbook info to statistics

        self.tick_counter += 1

        terminate: bool = self.should_stop or (self.tick_counter >= self.max_ticks)
        if terminate:
            HummingbotApplication.main_application().stop()

        if (self.tick_counter % 10) != 0:
            return

        self.estimate_params()

        active_helpers_list: List[Accumulator] = list(self.active_helpers)
        for helper in active_helpers_list:
            if RsState.STOP == helper.get_current_state():
                self.active_helpers.remove(helper)

                continue

            if not helper.is_started():
                if not helper.should_start(self.tick_counter):
                    continue

                helper.do_start(True)

            order_params = helper.execute_accumulation()

            if RsState.TRANSFORM_CALC == helper.get_current_state():
                helper.set_base_price(self.get_mean_base_price())

            elif order_params.side in ["buy", "sell"]:
                order_id = self.place_order(order_params)

                if "" == order_id:
                    self.logger().error("Could not place %s order for helper %s",
                                        order_params.side, helper.instance_name())
                    self.logger().info("Helper %s will be removed",
                                       helper.instance_name)

                    self.active_helpers.remove(helper)

                self.add_order(helper=helper, order_id=order_id)

    async def on_stop(self):
        """
        Is called when the `stop` is sent from the user interface
        """
        # TODO send status update to all helpers

        self.should_stop = True

    def did_fill_order(self, event: OrderFilledEvent):
        """
        Receive each event of full or partial order fill caused by this bot. It is then
        transferred to the accumulator helper that should handle it.
        """

        order_id: str = event.order_id
        if order_id not in self.active_orders:
            self.logger().error("Order filling %s not registered as active", order_id)

            return

        helper: Accumulator = self.active_orders[order_id]
        if helper not in self.active_helpers:
            self.logger().error("Helper %s not registered as active", helper.instance_name)

            return

        helper.did_fill_order(event)

    def did_fail_order(self, event: MarketOrderFailureEvent):
        """
        Receive each failed order event caused by this bot. It is then
        transferred to the accumulator helper that should handle it.

        The helper will go to the STOP state and will be taken off
        from the set of active states
        """

    def did_cancel_order(self, event: OrderCancelledEvent):
        """
        Receive each canceled order event caused by a third party.
        It is then transferred to the accumulator helper that should
        handle it.

        The helper will go to the STOP state and will be taken off
        from the set of active states
        """

    def did_expire_order(self, event: OrderExpiredEvent):
        """
        Receive each expired order event caused by a third party.
        It is then transferred to the accumulator helper that should
        handle it.

        The helper will go to the STOP state and will be taken off
        from the set of active states
        """

    def did_complete_buy_order(self, event: BuyOrderCompletedEvent):
        """
        Receive each complete buy order event. It is then transferred
        to the accumulator helper that issued the order.
        """

        order_id: str = event.order_id

        if order_id not in self.active_orders:
            self.logger().error("Order %s was not found in the list of expected orders",
                                order_id)

            # HINT return because there's nothing else to do
            return

        helper: Accumulator = self.active_orders[order_id]

        helper.did_complete_buy_order(event)

    def did_complete_sell_order(self, event: SellOrderCompletedEvent):
        """
        Receive each complete buy order event. It is then transferred
        to the accumulator helper that issued the order.
        """

    def add_helper(self, helper: Accumulator) -> bool:
        """
        Add a helper to the list of traders. Return True if addition was successful
        """

        if helper in self.active_helpers:
            self.logger().warning("Slave %s already in the list of active helpers",
                                  helper.instance_name())

            # HINT return to indicate failure
            return False

        self.active_helpers.add(helper)

        # HINT Normal function termination
        return True

    def add_order(self, helper: Accumulator, order_id: str) -> bool:
        """
        Add an order that a registered helper is waiting for
        """

        if helper in self.active_orders.values():
            self.logger().warning("Assistant %s has already a pending order %s",
                                  helper.instance_name, helper.get_active_order())

            # HINT return to indicate failure
            return False

        if order_id in self.active_orders:
            self.logger().warning("Order %s from helper %s is already in the list",
                                  helper.get_active_order(), helper.instance_name)

            # HINT return to indicate failure
            return False

        self.active_orders[order_id] = helper

        # Normal function termination
        return True

    def format_status(self) -> str:
        result: str = ""

        result += "Active trades\n"
        for accumlator in self.active_helpers:
            result += accumlator.instance_name() + "\n"

        # HINT Normal function termination
        return result
