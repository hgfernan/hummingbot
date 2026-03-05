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
import logging  # class Logger, getLogger()

# import json
import os
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel

from hummingbot.client.hummingbot_application import HummingbotApplication

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


class TransformParams(BaseModel):
    """
    Parameters for the RttState.TRANSFORM_CALC and RttState.TRANSFORM_ACTION states
    """
    investment: Decimal = Decimal(0.0)
    base_price: Decimal = Decimal(0.0)
    rel_delta: Decimal = Decimal(0.0)


class RestoreParams(BaseModel):
    """
    Parameters for the RttState.RESTORE_CALC and RttState.RESTORE_ACTION states
    """
    exchange_fee: Decimal = Decimal(0.0)
    gain_ratio: Decimal = Decimal(0.0)


class PriceAmount(BaseModel):
    """
    Base asset price and amount
    """
    amount: Decimal = Decimal(0.0)
    price: Decimal = Decimal(0.0)


class Accumulator(ABC):
    """
    Future abstract base class of the accumulation of base and quote assets.

    Currently will implement the accumulation of quote asset as a meeans to
    cover most cases before abstracting them in two specializedd classes,
    `BaseAccumulator` and `QuoteAccumulator`
    """

    last_instance_id: int = 0

    @classmethod
    def class_name(cls) -> str:
        """
        Name of the class
        """

        return type(cls).__name__

    # def __init__(self,
    #              investment: Decimal,
    #              base_price: Decimal,
    #              rel_delta: Decimal,
    #              exchange_fee: Decimal,
    #              gain_ratio: Decimal) -> None:
    def __init__(self,
                 transform_params: TransformParams,
                 restore_params: RestoreParams) -> None:
        """
        Initialize control variables and totalizers of gain and loss
        """

        # HINT Due to Python language each derived class have an independent `id`` numbering
        self.instance_id = self.__class__.inc_instance_id()

        self.curr_state = RttState.START

        self.curr_order = ""

        # HINT the price and amount for the current order
        self.curr_price_amount = PriceAmount()

        # HINT the price and amount of the partial fills of the last order
        self.partial_fills: List[Tuple[Decimal, Decimal]] = []

        # # HINT the amount available, either in base or quote asset
        # self.investment = investment

        # # HINT the base price to start the round trip cycle
        # self.base_price = base_price

        # # HINT the relative delta to adjust the base price
        # self.rel_delta = rel_delta

        # # HINT the maker exchange fee for the spot market
        # self.exchange_fee = exchange_fee

        # # HINT The ratio of the gain relatively to the exchange fee
        # self.gain_ratio = gain_ratio

        self.transform_params = transform_params

        self.restore_params = restore_params

    @classmethod
    def inc_instance_id(cls) -> int:
        """
        Increment the last instance id, and return the current one
        """
        result: int = cls.last_instance_id

        cls.last_instance_id += 1

        # HINT Normal function termination
        return result

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

    def get_curr_price_amount(self) -> PriceAmount:
        """
        Return the base price and amount of the last active order
        """

        return self.curr_price_amount

    def get_base_price(self) -> Decimal:
        """
        Return the current base price
        """

        return self.transform_params.base_price

    def get_rel_delta(self) -> Decimal:
        """
        Return the current relative delta ratio
        """

        return self.transform_params.rel_delta

    def set_base_price(self, base_price: Decimal) -> Decimal:
        """
        Set the base price to start the round trip trading cycle, and
        return the old base price
        """

        # HINT get the old price
        result: Decimal = self.transform_params.base_price

        self.transform_params.base_price = base_price

        # HINT Normal function termination
        return result

    def set_rel_delta(self, rel_delta: Decimal) -> Decimal:
        """
        Set the relative delta ratio to affect the base price and to
        start the round trip trading cycle, and return the old
        relative delta
        """

        result: Decimal = self.transform_params.rel_delta

        self.transform_params.rel_delta = rel_delta

        # HINT Normal function termination
        return result

    @abstractmethod
    def calc_transform_price(self) -> Decimal:
        """
        Calculate the transformation price, using base_price and rel_delta
        """

    @abstractmethod
    def calc_restore_price(self) -> Decimal:
        """
        Calculate the restoration price, using curr_price, exchange_fee and
        gain_ratio
        """

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

    last_instance_id: int = 0

    def __init__(self,
                 transform_params: TransformParams,
                 restore_params: RestoreParams) -> None:
        """
        Initialize a QuoteAccumulator instance
        """

        super().__init__(transform_params=transform_params, restore_params=restore_params)

        self.id: int = self.__class__.inc_instance_id()

        self.local_logger: Optional[logging.Logger] = None

    def logger(self) -> logging.Logger:
        """
        Return this class logger. Create one if necessary.
        """
        if self.local_logger is None:
            self.local_logger = logging.getLogger(self.__class__.__name__)

        # Normal function termination
        return self.local_logger

    def calc_transform_price(self) -> Decimal:
        """
        Calculate the transformation price, using base_price and rel_delta
        """
        result: Decimal = self.get_base_price() * (1 - self.transform_params.rel_delta)

        self.curr_price_amount.price = result

        # HINT Normal function termination
        return result

    def calc_restore_price(self) -> Decimal:
        """
        Calculate the restoration price, using curr_price, exchange_fee and
        gain_ratio
        """

        exchange_fee: Decimal = self.restore_params.exchange_fee
        gain_ratio: Decimal = self.restore_params.gain_ratio

        denom: float = 1.0 - float(exchange_fee)
        denom *= denom

        result: Decimal = Decimal(1.0) + gain_ratio * exchange_fee
        result /= Decimal(denom)

        # HINT Normal function termination
        return result

    def execute_accumulation(self) -> None:
        """
        Manage the finite state automaton for the accumulation, and keep
        a log of gain and loss
        """

        state: RttState = self.get_current_state()
        match state:
            case RttState.START:
                self.logger().info(state.name)

            case RttState.TRANSFORM_CALC:
                self.logger().info(state.name)

            case RttState.TRANSFORM_ACTION:
                self.logger().info(state.name)

            case RttState.RESTORE_CALC:
                self.logger().info(state.name)

            case RttState.RESTORE_ACTION:
                self.logger().info(state.name)

            case RttState.STOP:
                self.logger().info(state.name)

            case _:
                msg: str = ''

                if isinstance(state, RttState):
                    msg = f"Invalid or unknown state: {state.name} ({state.value})"

                else:
                    msg = f"Invalid or unknown state: {state}"

                self.logger().error(msg)

    def issue_transform_order(self) -> bool:
        """
        Issue a transform order
        """
        result: bool = True

        # HINT Normal function termination
        return result

    def issue_restore_order(self) -> bool:
        """
        Issue a restore order
        """
        result: bool = True

        # HINT Normal function termination
        return result

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

    def did_complete_buy_order(self, event: BuyOrderCompletedEvent):
        """
        If this was in a TRANSFORM_ACTION state it will go to RESTORE_CALC,
        and if it was in a RESTORE_ACTION it will go to the TRANSFORM_CALC
        state. It will go to the STOP state if it was currently waiting for
        a sell order but if this state arrived instead.
        """

    def did_complete_sell_order(self, event: SellOrderCompletedEvent):
        """
        If this was in a  it will go to RestoreCalculus, and if it was
        in a RESTORE_ACTION it will go either to the TransformCalculus.
        It will also go to the STOP state if it was currently waiting
        for a buy order but if this state arrived instead.
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

    # HINT number of QuoteAccumulator slaves
    n_quote_accumulators: int = 1

    # HINT number of BaseAccumulator slaves
    n_base_accumulators: int = 1

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
        self.active_orders: Dict[str, Accumulator] = {}

        self.tick_counter: int = 0

        self.estimate_params()

        # HINT allocate quote accumulators
        transform_params: TransformParams = \
            TransformParams(investment=self.quote_investment,
                            base_price=self.mean_base_price,
                            rel_delta=self.rel_delta)
        restore_params: RestoreParams = \
            RestoreParams(exchange_fee=self.fee, gain_ratio=self.gain_ratio)

        for qa_ind in range(self.n_quote_accumulators):
            qa: QuoteAccumulator = QuoteAccumulator(transform_params, restore_params)

            if not self.add_slave(qa):
                self.logger().error("FATAL Could not add QuoteAccumulator %d", qa_ind)

                HummingbotApplication.main_application().stop()

            self.active_slaves.add(qa)

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
        no more slaves to process, or while there are still iterations to run.
        """

        if (self.tick_counter % 10) == 0:
            self.estimate_params()

        self.tick_counter += 1

        terminate: bool = self.should_stop or (self.tick_counter >= self.max_ticks)
        if not terminate:
            active_slaves_list: List[Accumulator] = list(self.active_slaves)
            for qa in active_slaves_list:
                if RttState.STOP == qa.get_current_state():
                    self.active_slaves.remove(qa)

                    continue

                qa.execute_accumulation()

        HummingbotApplication.main_application().stop()

    async def on_stop(self):
        """
        Is called when the `stop` is sent from the user interface
        """
        # TODO send status update to all slaves

        self.should_stop = True

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
