"""
File: download_order_book.py
Created on 2024-06-18 19:36:51
Author: @hgfernan
Description: Script to download order book snapshots for specified trading pairs and exchange.
Obs: Adapted from download_order_book_and_trades.py, removing the trades related code and logic.
"""

import os.path # join()

import logging
import datetime # class datetime

from typing import Dict, Optional, Set
import pprint # pformat()

import pandas as pd
from pydantic import BaseModel # data manipulation library

from hummingbot import data_path
from hummingbot.connector.connector_base import ConnectorBase # base class for connectors
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.client.hummingbot_application import HummingbotApplication
from hummingbot.logger import HummingbotLogger
from hummingbot.client.settings import DEFAULT_LOG_FILE_PATH

LSB_LOGGER : Optional[logging.Logger] = None

def log_path() -> str:
    return DEFAULT_LOG_FILE_PATH

class DownloadOrderBookSnapshots(ScriptStrategyBase):
    """
    This strategy downloads order book snapshots for a specified trading pair
    and exchange at regular intervals and saves them to a CSV file. The
    strategy  collects a specified number of snapshots before saving them to
    a file and then stops the strategy. The file is named with a timestamp to
    ensure uniqueness.
    """
    exchange : str = "binance_paper_trade" # HINT default exchange
    trading_pair : str = "ETH-USDT" # default trading pair
    depth : int = 50 # HINT default depth for order book snapshots
    dump_interval : int = 10 # HINT time interval between dumps in seconds
    n_dumps : int = 6 # HINT number of dumps before creating a new file
    dump_count : int = 0 # HINT counter for the number of dumps
    markets : Dict[str, Set[str]] = {exchange: set([trading_pair])}


    # HINT temporary storage for order book snapshots
    ob_temp_storage : pd.DataFrame = pd.DataFrame()

    # HINT current date for file naming
    start_ts : datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)
    last_ts : datetime.datetime = start_ts

    # HINT file path for storing order book snapshots
    ob_file_path : str = \
        f"{data_path()}/orderbook_{start_ts.strftime('%Y%m%d_%H%M%S')}.csv"

    log_level : int = logging.INFO
    save_log : bool = True

    @classmethod
    def logger(cls) -> HummingbotLogger:

        global LSB_LOGGER
        log_format : str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        if LSB_LOGGER is not None:
            return LSB_LOGGER

        LSB_LOGGER = logging.getLogger(__name__)
        LSB_LOGGER.setLevel(cls.log_level)
        formatter : logging.Formatter = logging.Formatter(log_format)

        ch = logging.StreamHandler()
        ch.setLevel(cls.log_level)
        ch.setFormatter(formatter)

        LSB_LOGGER.addHandler(ch)

        if cls.save_log:
            log_file_path = os.path.join(log_path(), __name__ + '.log')

            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(cls.log_level)
            file_handler.setFormatter(formatter)

            LSB_LOGGER.addHandler(file_handler)
        

        return LSB_LOGGER

    @classmethod
    def init_markets(cls, config: BaseModel) -> None:
        """
        Initialize the markets for the specified exchange and trading pair.
        This method is called before the strategy starts and is used to set
        up the necessary market data for the strategy to function properly.
        """
        cls.connectors[cls.exchange].add_trading_pair(cls.trading_pair)

        cls.logger().info("Initialized markets for %s and %s",
                          cls.exchange, cls.trading_pair)

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)

    def on_tick(self) -> None:
        # HINT calculate the time difference in seconds
        current_ts : datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)
        sec_diff : int = round((current_ts - self.last_ts).total_seconds())

        # HINT return if time difference is less than the dump interval
        if sec_diff < self.dump_interval:
            return

        # HINT check if the number of dumps has reached the limit
        if self.dump_count < self.n_dumps:
            order_book_data : Dict[str, object] = \
                self.get_order_book_dict(self.exchange, self.trading_pair, self.depth)

            if not order_book_data["bids"] and not order_book_data["asks"]:
                self.logger().warning("No order book data available for %s on %s",
                                      self.trading_pair, self.exchange)
                return

            if self.dump_count == 0:
                self.logger().info("Starting to collect order book snapshots for %s on %s",
                                   self.trading_pair, self.exchange)
                # self.logger().info(pprint.pformat(dir(order_book_data)))

            # HINT append the order book data to the temporary storage
            self.ob_temp_storage = \
                pd.concat([self.ob_temp_storage, pd.DataFrame([order_book_data])],
                          ignore_index=True)

            self.logger().info("Collected order book snapshot %d for %s on %s",
                               self.dump_count + 1, self.trading_pair, self.exchange)

            self.dump_count += 1
            self.last_ts = current_ts

            # HINT go to the next tick after the specified dump interval
            return

        # HINT Save the order book snapshots to the file
        self.create_obook_file() # create new files

        msg : str = f"Will stop the strategy after saving {self.n_dumps} dumps to the file."
        self.logger().info(msg)

        # HINT leave the strategy after saving the order book snapshots
        HummingbotApplication.main_application().stop()

    def get_order_book_dict(self,
                            exchange: str,
                            trading_pair: str,
                            depth: int = 50) -> Dict[str, object]:
        """
        Get the order book snapshot for the specified exchange, trading pair, and depth.
        """
        # HINT get the order book from the connector
        order_book = self.connectors[exchange].get_order_book(trading_pair)

        # HINT get the snapshot of the order book
        snapshot = order_book.snapshot

        if self.dump_count == 0:
            self.logger().info("Order book directory:\n%s",
                                pprint.pformat(dir(order_book)) )
            self.logger().info("Order book snapshot directory:\n%s",
                                pprint.pformat(dir(snapshot)) )

        # HINT format the snapshot of the order book as a dictionary
        snapshot = order_book.snapshot
        return {
            # HINT timestamp of the snapshot
            "ts": self.current_timestamp,

            # HINT list of bids up to the specified depth
            "bids": snapshot[0].loc[:(depth - 1), ["price", "amount"]].values.tolist(),

            # HINT list of asks up to the specified depth
            "asks": snapshot[1].loc[:(depth - 1), ["price", "amount"]].values.tolist(),
        }

    def create_obook_file(self) -> None:
        """
        Save the temporary storage to a CSV file
        """
        df : pd.DataFrame = self.ob_temp_storage.explode('bids', ignore_index=True)
        df = df.explode('asks', ignore_index=True)
        df.to_csv(self.ob_file_path, index=False)

        msg : str = f"Order book snapshots saved to {self.ob_file_path}"
        self.logger().info(msg)
