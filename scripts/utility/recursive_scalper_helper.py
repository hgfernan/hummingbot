#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: recursive_scalper.py
Created: 2026-03030 17:22:58
@author: @hgfernan
Description: Implement the Round Trip Trading algorithm
"""

# from datetime import datetime
import enum  # class Enum, auto()
import logging  # class Logger, getLogger()

# import json
# import os
from abc import ABC, abstractmethod
from decimal import Decimal

# from typing import Any, Dict, List, Optional, Set
from typing import List, Optional

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
    OrderType,
    PositionAction,
    SellOrderCompletedEvent,
)

# from hummingbot.client.hummingbot_application import HummingbotApplication

# from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class RsState(enum.Enum):
    """
    Enumeration of Round Trip Trading states
    """
    START = 0
    TRANSFORM_CALC = enum.auto()
    TRANSFORM_ACTION = enum.auto()
    RESTORE_CALC = enum.auto()
    RESTORE_ACTION = enum.auto()
    STOP = enum.auto()

    def get_description(self) -> str:
        """
        Mapping of RsState values to descriptions
        """
        result: str = ""
        match self:
            case self.START:
                result = "Processing Start"

            case self.TRANSFORM_CALC:
                result = "Transformation Calculus"

            case self.TRANSFORM_ACTION:
                result = "Transformation Action"

            case self.RESTORE_CALC:
                result = "Restoration Calculus"

            case self.RESTORE_ACTION:
                result = "Restoration Action"

            case self.STOP:
                result = "Processing Stop"

            case _:
                result = f"Unknown or invalid state: {self}"

        # HINT Normal function termination
        return result


class ActivityParams(BaseModel):
    """
    Main parameters for activity
    """
    started: bool = False
    curr_order: str = ""
    curr_state: RsState = RsState.START
    instance_id: int = 0
    starting_tick: int = 0


class TransformParams(BaseModel):
    """
    Parameters for the RsState.TRANSFORM_CALC and RsState.TRANSFORM_ACTION states
    """
    investment: Decimal = Decimal(0.0)
    base_price: Decimal = Decimal(0.0)
    rel_delta: Decimal = Decimal(0.0)


class RestoreParams(BaseModel):
    """
    Parameters for the RsState.RESTORE_CALC and RsState.RESTORE_ACTION states
    """
    exchange_fee: Decimal = Decimal(0.0)
    gain_ratio: Decimal = Decimal(0.0)


class PriceAmount(BaseModel):
    """
    Base asset price and amount
    """
    amount: Decimal = Decimal(0.0)
    price: Decimal = Decimal(0.0)

# OrderParams = Dict[str, Union[str, Decimal, OrderType, PositionAction]]


class OrderParams(BaseModel):
    """
    All parameters for order placing RoundTripTrading methods `buy()` and `sell()`
    """
    side: str = ""
    state_name: str = ""
    connector_name: str = ""
    trading_pair: str = ""
    amount: Decimal = Decimal(0)
    order_type: OrderType = OrderType.LIMIT
    price: Decimal = Decimal(0)
    position_action: PositionAction = PositionAction.OPEN


class Accumulator(ABC):
    """
    Abstract base class for the accumulation of base and quote assets.

    Currently will implement the accumulation of quote asset as a meeans to
    cover most cases before abstracting them in two specializedd classes,
    `BaseAccumulator` and `QuoteAccumulator`
    """

    last_instance_id: int = 0

    def _class_name(self) -> str:
        """
        Name of the class
        """

        cls = self.__class__
        return type(cls).__name__

    def __init__(self,
                 starting_tick: int,
                 transform_params: TransformParams,
                 restore_params: RestoreParams) -> None:
        """
        Initialize control variables and totalizers of gain and loss
        """

        self.activity_params: ActivityParams = \
            ActivityParams(starting_tick=starting_tick,
                           instance_id=self.__class__._inc_instance_id())

        # # HINT Each derived class have an independent `id`` numbering
        # self.instance_id =

        # self.started: bool = False
        # self.curr_state: RsState = RsState.START

        # self.curr_order = ""

        # HINT the price and amount issued for the current order
        self.curr_price_amount = PriceAmount()

        # HINT the price and amount of the partial fills of the last order
        self.partial_fills: List[PriceAmount] = []

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
    def _inc_instance_id(cls) -> int:
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
        return self._class_name() + "-" + f"{self.activity_params.instance_id:02d}"

    def is_started(self) -> bool:
        """
        Was this instance already started ?
        """

        return self.activity_params.started

    def get_current_state(self) -> RsState:
        """
        Return the current state
        """

        return self.activity_params.curr_state

    def get_active_order(self) -> str:
        """
        Return the last issued order, before the issuing of another one.
        """

        return self.activity_params.curr_order

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

    def should_start(self, tick_counter: int) -> bool:
        """
        Return True if the helper should start
        """

        return self.activity_params.starting_tick >= tick_counter

    def do_start(self, start: bool = True) -> bool:
        """
        Start this instance, and return the previous started
        """

        result: bool = self.is_started()

        self.activity_params.started = start

        return result

    def get_rel_delta(self) -> Decimal:
        """
        Return the current relative delta ratio
        """

        return self.transform_params.rel_delta

    def set_current_state(self, next_state: RsState) -> RsState:
        """
        Set the next current state
        """
        result: RsState = self.get_current_state()

        self.activity_params.curr_state = next_state

        return result

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
    def execute_accumulation(self) -> OrderParams:
        """
        Manage the finite state automaton for the accumulation, and keep
        a log of gain and loss
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
        This helper will go to the STOP state if it receives this message, and
        will stop further processing, and won't be seen by the master anymore.
        """

    def did_cancel_order(self, event: OrderCancelledEvent):
        """
        This helper will go to the STOP state if it receives this message, and
        will stop further processing, and won't be seen by the master anymore.
        """

    def did_expire_order(self, event: OrderExpiredEvent):
        """
        This helper will go to the STOP state if it receives this message, and
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

    def mandatory_stop(self) -> RsState:
        """
        Cause the helper to go to the STOP state
        """

        result: RsState = self.activity_params.curr_state

        self.activity_params.curr_state = RsState.STOP

        # HINT Normal function termination
        return result


class QuoteAccumulator(Accumulator):
    """
    Accumulator of the quote asset
    """

    last_instance_id: int = 0

    def __init__(self,
                 starting_tick: int,
                 transform_params: TransformParams,
                 restore_params: RestoreParams) -> None:
        """
        Initialize a QuoteAccumulator instance
        """

        super().__init__(starting_tick,
                         transform_params=transform_params,
                         restore_params=restore_params)

        self.id: int = self.__class__._inc_instance_id()

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

    def execute_accumulation(self) -> OrderParams:
        """
        Manage the finite state automaton for the accumulation, and keep
        a log of gain and loss
        """

        result: OrderParams = OrderParams()

        state: RsState = self.get_current_state()
        match state:
            case RsState.START:
                self.logger().info(state.name)

                result.state_name = state.get_description()

                # TODO what else should be done here ?

                self.set_current_state(RsState.TRANSFORM_ACTION)

            case RsState.TRANSFORM_CALC:
                self.logger().info(state.name)

                result.state_name = state.get_description()

                # TODO calculate price and format order

                self.set_current_state(RsState.TRANSFORM_ACTION)

            case RsState.TRANSFORM_ACTION:
                self.logger().info(state.name)

                result.side = "buy"
                result.state_name = state.get_description()

                # TODO wait here till buy order is complete
                # HINT the state change will be caused by did_complete_buy_order()

            case RsState.RESTORE_CALC:
                self.logger().info(state.name)

                result.state_name = state.get_description()

            case RsState.RESTORE_ACTION:
                self.logger().info(state.name)

                result.side = "sell"
                result.state_name = state.get_description()

            case RsState.STOP:
                self.logger().info(state.name)

                result.state_name = state.get_description()

            case _:
                msg: str = ''

                if isinstance(state, RsState):
                    msg = f"Invalid or unknown state: {state.name} ({state.value})"

                else:
                    msg = f"Invalid or unknown state: {state}"

                self.logger().error(msg)

                result.state_name = msg

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

        self.partial_fills.append(PriceAmount(price=event.price, amount=event.amount))

    def did_fail_order(self, event: MarketOrderFailureEvent):
        """
        This helper will go to the STOP state if it receives this message, and
        will stop further processing, and won't be seen by the master anymore.
        """

    def did_cancel_order(self, event: OrderCancelledEvent):
        """
        This helper will go to the STOP state if it receives this message, and
        will stop further processing, and won't be seen by the master anymore.
        """

    def did_expire_order(self, event: OrderExpiredEvent):
        """
        This helper will go to the STOP state if it receives this message, and
        will stop further processing, and won't be seen by the master anymore.
        """

    def did_complete_buy_order(self, event: BuyOrderCompletedEvent):
        """
        A `QuoteAccumulator` in `RsState.TRANSFORM_ACTION` will go to
        `RsState.RESTORE_CALC`; a `BaseAccumulator` in `RsState.RESTORE_ACTION`
        will go to `RsState.TRANSFORM_CALC`
        """

        self.set_current_state(RsState.RESTORE_CALC)

    def did_complete_sell_order(self, event: SellOrderCompletedEvent):
        """
        A `QuoteAccumulator` in `RsState.RESTORE_ACTION` will go to
        `RsState.TRANSFORM_CALC`; a `BaseAccumulator` in `RsState.TRANSFORM_ACTION`
        will go to `RsState.RESTORE_CALC`
        """

        self.set_current_state(RsState.RESTORE_CALC)
