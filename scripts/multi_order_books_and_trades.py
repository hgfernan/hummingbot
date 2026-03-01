#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: multi_order_books_and_trades.py
Description: Several downloads of order book and trades
Created: 2026-03-01 15:05
@author: Hummingbot Team
Derived from: download_order_book_and_trades.py
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel

from hummingbot import data_path
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.event.event_forwarder import SourceInfoEventForwarder
from hummingbot.core.event.events import OrderBookEvent, OrderBookTradeEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class MultiOrderBooksAndTrades(ScriptStrategyBase):
    """
    This script downloads mulitple order book and trades data for one trading pair.
    """

    exchange = os.getenv("EXCHANGE", "binance_paper_trade")
    trading_pairs = os.getenv("TRADING_PAIRS", "ETH-USDT")
    trading_pairs = list(trading_pairs.split(","))

    depth: Any = os.getenv("DEPTH", "50")
    if isinstance(depth, str) and depth.isdigit():
        depth = int(depth)
    else:
        raise ValueError("DEPTH must be an integer")

    last_dump_timestamp = 0
    time_between_csv_dumps = 10

    ob_temp_storage = {trading_pair: [] for trading_pair in trading_pairs}
    trades_temp_storage = {trading_pair: [] for trading_pair in trading_pairs}
    current_date = None
    ob_file_paths = {}
    trades_file_paths = {}
    markets = {exchange: set(trading_pairs)}
    subscribed_to_order_book_trade_event: bool = False

    @classmethod
    def init_markets(cls, config: BaseModel):
        """
        This method is called in the start command if the script has a config class defined,
        and allows script to define the market connectors and trading pairs needed for the
        strategy operation.
        """
        cls.markets = {cls.exchange: set(cls.trading_pairs)}

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        """
        Initialize the class strategy instance.
        """
        super().__init__(connectors)
        self.create_order_book_and_trade_files()
        self.order_book_trade_event = SourceInfoEventForwarder(self._process_public_trade)

    def on_tick(self):
        """
        The on_tick method is called every second and is used to get the order book data and
        store it in a temporary storage.
        It also checks if it's time to dump the data to the file and clean the temporary
        storage
        """

        if not self.subscribed_to_order_book_trade_event:
            self.subscribe_to_order_book_trade_event()
        self.check_and_replace_files()
        for trading_pair in self.trading_pairs:
            order_book_data = self.get_order_book_dict(self.exchange, trading_pair, self.depth)
            self.ob_temp_storage[trading_pair].append(order_book_data)
        if self.last_dump_timestamp < self.current_timestamp:
            self.dump_and_clean_temp_storage()

    def get_order_book_dict(self, exchange: str, trading_pair: str, depth: int = 50):
        """
        Return a dictionary representation of the order book snapshot for a given trading pair.
        :param exchange: The name of the exchange to get the order book from.
        :param trading_pair: The trading pair to get the order book for.
        :param depth: The number of levels to include in the order book snapshot.
        :return:
            A dictionary containing the timestamp, bids, and asks of the order book snapshot.
        """

        order_book = self.connectors[exchange].get_order_book(trading_pair)
        snapshot = order_book.snapshot
        return {
            "ts": self.current_timestamp,
            "bids": snapshot[0].loc[: (depth - 1), ["price", "amount"]].values.tolist(),
            "asks": snapshot[1].loc[: (depth - 1), ["price", "amount"]].values.tolist(),
        }

    def dump_and_clean_temp_storage(self):
        """
        Dump the data in the temporary storage to the file and clean the temporary storage.
        """
        for trading_pair, order_book_info in self.ob_temp_storage.items():
            file = self.ob_file_paths[trading_pair]
            json_strings = [json.dumps(obj) for obj in order_book_info]
            json_data = "\n".join(json_strings)
            file.write("\n" + json_data)
            self.ob_temp_storage[trading_pair] = []

        for trading_pair, trades_info in self.trades_temp_storage.items():
            file = self.trades_file_paths[trading_pair]
            json_strings = [json.dumps(obj) for obj in trades_info]
            json_data = "\n".join(json_strings)
            file.write("\n" + json_data)
            self.trades_temp_storage[trading_pair] = []

        self.last_dump_timestamp = self.current_timestamp + self.time_between_csv_dumps

    def check_and_replace_files(self):
        """
        Check if the date has changed and replace the files if it has.
        """

        current_date = datetime.now().strftime("%Y-%m-%d")
        if current_date != self.current_date:
            for file in self.ob_file_paths.values():
                file.close()
            self.create_order_book_and_trade_files()

    def create_order_book_and_trade_files(self):
        """'
        Create the order book and trade files for each trading pair.
        """

        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.ob_file_paths = {
            trading_pair: self.get_file(self.exchange,
                                        trading_pair,
                                        "order_book_snapshots",
                                        self.current_date)
            for trading_pair in self.trading_pairs
        }

        self.trades_file_paths = {
            trading_pair: self.get_file(self.exchange,
                                        trading_pair,
                                        "trades",
                                        self.current_date)
            for trading_pair in self.trading_pairs
        }

    @staticmethod
    def get_file(exchange: str, trading_pair: str, source_type: str, current_date: str):
        """
        Get the file path for the order book or trades data and return the file object.

        :param exchange: The name of the exchange.
        :param trading_pair: The trading pair.
        :param source_type: The type of data (order book snapshots or trades).
        :param current_date: The current date in the format YYYY-MM-DD.

        :return: The file object for the order book or trades data.
        """

        file_path = data_path() + f"/{exchange}_{trading_pair}_{source_type}_{current_date}.txt"

        return open(file_path, "a", encoding="utf-8")

    def _process_public_trade(self, event_tag: int, market: ConnectorBase, event: OrderBookTradeEvent):
        """
        Process the public trade event and store the trade data in the temporary storage.

        :param event_tag: The event tag.
        :param market: The market connector that emitted the event.
        :param event: The OrderBookTradeEvent containing the trade data.
        """
        print(f"{event_tag=}, {market=}, {event=}")
        self.trades_temp_storage[event.trading_pair].append(
            {
                "ts": event.timestamp,
                "price": event.price,
                "q_base": event.amount,
                "side": event.type.name.lower(),
            }
        )

    def subscribe_to_order_book_trade_event(self):
        """
        Subscribe to the OrderBookTradeEvent for each order book of each market connector.
        """

        for market in self.connectors.values():
            for order_book in market.order_books.values():
                order_book.add_listener(OrderBookEvent.TradeEvent, self.order_book_trade_event)

        self.subscribed_to_order_book_trade_event = True
