#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: round_trip_trading.py
Created: 2026-03030 17:22:58
@author: @hgfernan
Description: Implement the Round Trip Trading algorithm
"""

# from datetime import datetime
import enum  # class Enum, auto()

# import json
import os
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Dict, Optional, Set, Tuple

from pydantic import BaseModel

# from hummingbot import data_path
# from hummingbot.connector.connector_base import ConnectorBase
# from hummingbot.core.event.event_forwarder import SourceInfoEventForwarder
from hummingbot.core.event.events import (  # BuyOrderCreatedEvent,; SellOrderCreatedEvent,; OrderBookEvent,; OrderBookTradeEvent,
    BuyOrderCompletedEvent,
    MarketOrderFailureEvent,
    OrderCancelledEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    SellOrderCompletedEvent,
)
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class RttState(enum.Enum):
    """
    Enumeration of Round Trip Trading states
    """
    START = 0
    TRANSFORM_CALC = enum.auto()
    TRANSFORM_ACTION = enum.auto()
    RESTORE_CALC = enum.auto()
    RESTORE_ACTION = enum.auto()
    STOP = enum.auto()


state_full_name: Dict[RttState, str] = {
    RttState.START: "Processing Start",
    RttState.TRANSFORM_CALC: "Transformation Calculus",
    RttState.TRANSFORM_ACTION: "Transformation Action",
    RttState.RESTORE_CALC: "Restoration Calculus",
    RttState.RESTORE_ACTION: "Restoration Action",
    RttState.STOP: "Processing Stop"
}


class Accumulator(ABC):
    """
    Future abstract base class of the accumulation of base and quote assets.

    Currently will implement the accumulation of quote asset as a meeans to
    cover most cases before abstracting them in two specializedd classes,
    `BaseAccumulator` and `QuoteAccumulator`
    """

    last_instance_id: int = 0

    @classmethod
    def inc_instance_id(cls) -> int:
        """
        Increment the instance id, and return the current one
        """
        result: int = cls.last_instance_id

        cls.last_instance_id += 1

        # HINT Normal function termination
        return result

    @classmethod
    def class_name(cls) -> str:
        """
        Name of the class
        """

        return type(cls).__name__

    def __init__(self):
        """
        Initialize control variables and totalizers of gain and loss
        """

        self.instance_id = self.__class__.inc_instance_id()

        self.curr_state = RttState.START

        self.curr_order = ""

        # HINT the price and amount for the current order
        self.curr_price_amount = None

        # HINT the price and amount of the partial fills of the last order
        self.partial_fills = []

        # HINT the delta to adjust the base price
        self.delta = Decimal(0.0)

        # HINT the base price to start the round trip cycle
        self.base_price = Decimal(0.0)

    def instance_name(self) -> str:
        """
        Name of the instance
        """
        cls = self.__class__

        return cls.class_name() + "-" + f"{self.instance_id:02d}"

    def get_current_state(self) -> RttState:
        """
        Return the current state
        """

        return self.curr_state

    def get_active_order(self) -> str:
        """
        Return the last issued order, before the issuing of another one.
        """

        return self.curr_order

    def get_curr_price_amount(self) -> Optional[Tuple[Decimal, Decimal]]:
        """
        Return the base price and amount of the last active order
        """

        return self.curr_price_amount

    def get_base_price(self) -> Decimal:
        """
        Return the current base price
        """

        return self.base_price

    def get_delta(self) -> Decimal:
        """
        Return the current delta ratio
        """

        return self.delta

    def set_base_price(self, base_price: Decimal) -> Decimal:
        """
        Set the base price to start the round trip trading cycle, and
        return the old base price
        """

        # HINT get the old price
        result: Decimal = self.base_price

        self.base_price = base_price

        # HINT Normal function termination
        return result

    def set_delta(self, delta: Decimal) -> Decimal:
        """
        Set the delta ratio to affect the base price and to start
        the round trip trading cycle, and return the old delta
        """

        result: Decimal = self.delta

        self.delta = delta

        # HINT Normal function termination
        return result

    @abstractmethod
    def execute_accumulation(self) -> None:
        """
        Manage the finite state automaton for the accumulation, and keep
        a log of gain and loss
        """

    @abstractmethod
    def issue_transform_order(self) -> bool:
        """
        Issue a transform order
        """

    @abstractmethod
    def issue_restore_order(self) -> bool:
        """
        Issue a restore order
        """

    # def did_create_buy_order(self, event: BuyOrderCreatedEvent):
        # """
        # Future versions will confirm a buy order was really issued
        # """
        #
        # pass

    # def did_create_sell_order(self, event: SellOrderCreatedEvent):
        # """
        # Future versions will confirm a sell order was really issued
        # """
        #
        # pass

    @abstractmethod
    def did_fill_order(self, event: OrderFilledEvent):
        """
        Register partial order fills, till the order fullfilling.

        Won't issue a complete order event.
        """

    def did_fail_order(self, event: MarketOrderFailureEvent):
        """
        This slave will go to the STOP state if it receives this message, and
        will stop further processing, and won't be seen by the master anymore.
        """

    def did_cancel_order(self, event: OrderCancelledEvent):
        """
        This slave will go to the STOP state if it receives this message, and
        will stop further processing, and won't be seen by the master anymore.
        """

    def did_expire_order(self, event: OrderExpiredEvent):
        """
        This slave will go to the STOP state if it receives this message, and
        will stop further processing, and won't be seen by the master anymore.
        """

    @abstractmethod
    def did_complete_buy_order(self, event: BuyOrderCompletedEvent):
        """
        If this was in a TRANSFORM_ACTION state it will go to RESTORE_CALC,
        and if it was in a RESTORE_ACTION it will go to the TRANSFORM_CALC
        state. It will go to the STOP state if it was currently waiting for
        a sell order but if this state arrived instead.
        """

    @abstractmethod
    def did_complete_sell_order(self, event: SellOrderCompletedEvent):
        """
        If this was in a  it will go to RestoreCalculus, and if it was
        in a RESTORE_ACTION it will go either to the TransformCalculus.
        It will also go to the STOP state if it was currently waiting
        for a buy order but if this state arrived instead.
        """

    def mandatory_stop(self) -> RttState:
        """
        Cause the slave to go to the STOP state
        """

        result: RttState = self.curr_state

        self.curr_state = RttState.STOP

        # HINT Normal function termination
        return result


class QuoteAccumulator(Accumulator):
    """
    Accumulator of the quote asset
    """


class RoundTripTrading(ScriptStrategyBase):
    """
    Manage the creation of accumulators, according to configuration, and their
    execution
    """

    exchange = os.getenv("EXCHANGE", "binance_paper_trade")
    trading_pair = os.getenv("TRADING_PAIRS", "ETH-USDT")

    depth: Any = os.getenv("DEPTH", "50")
    if isinstance(depth, str) and depth.isdigit():
        depth = int(depth)
    else:
        raise ValueError("DEPTH must be an integer")

    last_dump_timestamp = 0
    time_between_csv_dumps = 10

    obook_temp_storage = trading_pair
    trades_temp_storage = trading_pair
    current_date = None
    market = {exchange: trading_pair}
    subscribed_to_order_book_trade_event: bool = False

    quote_quantum: Decimal = Decimal(25.0)
    quote_ratio: Decimal = Decimal(0.05)

    base_quantum: Decimal = Decimal(25.0 / 2000)
    quote_ratio: Decimal = Decimal(0.05)

    @classmethod
    def init_markets(cls, config: BaseModel):
        """
        This method is called in the start command if the script has a config class defined,
        and allows script to define the market connectors and trading pairs needed for the
        strategy operation.
        """
        cls.markets = {cls.exchange: set(cls.trading_pairs)}

    def __init__(self, connectors: Dict[str, Any], config: Optional[BaseModel] = None):
        """
        Initialization of the RoundTripTrading instance.

        In future versions, parameters will be received a from configuration object.
        """

        super().__init__(connectors, config)

        # HINT set of active Accumulator objects
        self.active_slaves: Set[Accumulator] = set()

        # HINT dictionary of active orders for Accumulator objects
        self.active_orders = {}

        self.tick_counter: int = 0

    def estimate_params(self) -> None:
        """
        Estimate statistical parameters -- mostly the base price and the delta to
        adjust the base price
        """

    def on_tick(self) -> None:
        """
        Main program of the strategy: will receive Hummingbot ticks till there are
        no more slaves to process, or while there are still iterations to run.
        """

        if (self.tick_counter % 10) == 0:
            self.estimate_params()

        if self.tick_counter == 0:
            # TODO allocate slaves
            pass

        self.tick_counter += 1

    def did_fill_order(self, event: OrderFilledEvent):
        """
        Receive each event of full or partial order fill caused by this bot. It is then
        transferred to the accumulator slave that should handle it.
        """

    def did_fail_order(self, event: MarketOrderFailureEvent):
        """
        Receive each failed order event caused by this bot. It is then
        transferred to the accumulator slave that should handle it.

        The slave will go to the STOP state and will be taken off
        from the set of active states
        """

    def did_cancel_order(self, event: OrderCancelledEvent):
        """
        Receive each canceled order event caused by a third party.
        It is then transferred to the accumulator slave that should
        handle it.

        The slave will go to the STOP state and will be taken off
        from the set of active states
        """

    def did_expire_order(self, event: OrderExpiredEvent):
        """
        Receive each expired order event caused by a third party.
        It is then transferred to the accumulator slave that should
        handle it.

        The slave will go to the STOP state and will be taken off
        from the set of active states
        """

    def did_complete_buy_order(self, event: BuyOrderCompletedEvent):
        """
        Receive each complete buy order event. It is then transferred
        to the accumulator slave that issued the order.
        """

    def did_complete_sell_order(self, event: SellOrderCompletedEvent):
        """
        Receive each complete buy order event. It is then transferred
        to the accumulator slave that issued the order.
        """

    def add_slave(self, slave: Accumulator) -> bool:
        """
        Add a slave to the list of traders. Return True if addition was successful
        """

        if slave in self.active_slaves:
            self.logger().warning("Slave %s already in the list of active slaves",
                                  slave.instance_name())

            # HINT return to indicate failure
            return False

        self.active_slaves.add(slave)

        # HINT Normal function termination
        return True

    def add_order(self, slave: Accumulator, order_id: str) -> bool:
        """
        Add an order that a registered slave is waiting for
        """

        if slave in self.active_orders.values():
            self.logger().warning("Slave %s has already a pending order %s",
                                  slave.instance_name, slave.get_active_order())

            # HINT return to indicate failure
            return False

        if order_id in self.active_orders:
            self.logger().warning("Order %s from slave %s is already in the list",
                                  slave.get_active_order(), slave.instance_name)

            # HINT return to indicate failure
            return False

        self.active_orders[order_id] = slave

        # Normal function termination
        return True
