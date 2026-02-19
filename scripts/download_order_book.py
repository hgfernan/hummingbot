# File: download_order_book.py
# Created on 2024-06-18 19:36:51
# Author: @hgfernan
# Description: Script to download order book snapshots for specified trading pairs and exchange.

import json # loads(), dumps()
import datetime # class datetime

from typing import Dict, List, Tuple # type hinting for dictionaries and lists

import pandas as pd # data manipulation library

from hummingbot import data_path
from hummingbot.connector.connector_base import ConnectorBase # base class for connectors
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.client.hummingbot_application import HummingbotApplication


class DownloadOrderBookSnapshots(ScriptStrategyBase):
    exchange : str = "binance_paper_trade" # default exchange
    trading_pair : str = "ETH-USDT" # default trading pair
    depth : int = 50 # default depth for order book snapshots
    dump_interval : int = 10 # HINT time interval between dumps in seconds
    n_dumps : int = 6 # HINT number of dumps before creating a new file
    dump_count : int = 0 # HINT counter for the number of dumps

    # HINT temporary storage for order book snapshots
    ob_temp_storage : pd.DataFrame = pd.DataFrame() 

    # HINT current date for file naming
    start_ts : datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)
    last_ts : datetime.datetime = start_ts

    # HINT file path for storing order book snapshots  
    ob_file_path : str = f"{data_path()}/orderbook_{start_ts.strftime('%Y%m%d_%H%M%S')}.csv" 

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)

    def on_tick(self):
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

            # TODO append the order book data to the temporary storage
            self.ob_temp_storage = \
                pd.concat([self.ob_temp_storage, pd.DataFrame([order_book_data])], 
                          ignore_index=True)

            self.dump_count += 1
            self.last_ts = current_ts

            # HINT go to the next tick after the specified dump interval
            return

        # HINT Save the order book snapshots to the file
        self.create_obook_file() # create new files

        # HINT leave the strategy after saving the order book snapshots
        HummingbotApplication.main_application().stop()

    def get_order_book_dict(self, exchange: str, trading_pair: str, depth: int = 50):
        # HINT get the order book from the connector
        order_book = self.connectors[exchange].get_order_book(trading_pair) 

        # HINT get the snapshot of the order book and format it as a dictionary
        snapshot = order_book.snapshot 
        return {
            "ts": self.current_timestamp, # timestamp of the snapshot
            "bids": snapshot[0].loc[:(depth - 1), ["price", "amount"]].values.tolist(), # list of bids up to the specified depth
            "asks": snapshot[1].loc[:(depth - 1), ["price", "amount"]].values.tolist(), # list of asks up to the specified depth
        }

    def create_obook_file(self):
        # HINT save the temporary storage to a CSV file
        self.ob_temp_storage.to_csv(self.ob_file_path, index=False) 
        self.logger().info(f"Order book snapshots saved to {self.ob_file_path}")    
