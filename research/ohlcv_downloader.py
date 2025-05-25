import os
from pathlib import Path

from dotenv import load_dotenv,find_dotenv
import pandas as pd

from sbt_quant_helpers.data.utils import get_db_connection
from sbt_quant_helpers.data.utils import get_hermes_instance
from sbt_quant_helpers.data.utils import get_ohlcv_from_hdp
from sbt_quant_helpers.data.utils import get_all_assets

from svcj.config import DATA_DIR, IS_START, IS_END, MONGO_URI

SANDBOX_DB = get_db_connection(MONGO_URI,db_name='Sandbox')
GARAGE_DB = get_db_connection(MONGO_URI,db_name='Garage')

DATA_DIR = Path(DATA_DIR)

START_DATE = pd.to_datetime(IS_START, utc=True)
END_DATE = pd.to_datetime(IS_END, utc=True)

ASSET_FILTERING_CONFIG = {
    'filter_stablecoins': True,
    'filter_delisted_assets': False,
    'filter_leverage_tokens': True,
    'filter_start_date': False,
}

hm = get_hermes_instance()
logger = hm.logger

loop_list = [
    ('BINANCE', 'PERP', 'USDT'),
]

for exchange, market_type, quote_asset in loop_list:

    logger.info(f'Downloading data for {exchange}_{market_type}: {quote_asset} PAIRS')

    assets = get_all_assets(db = SANDBOX_DB, asset_filtering_config = ASSET_FILTERING_CONFIG, start_date = START_DATE,end_date = END_DATE, exchange=exchange, market_type=market_type, quote_asset=quote_asset)

    get_ohlcv_from_hdp(
        data_path = DATA_DIR,
        db = GARAGE_DB,
        hm = hm,
        logger = logger,
        start_date = START_DATE,
        filtering_end_date = END_DATE,
        end_date = END_DATE,
        exchange = exchange,
        market_type = market_type,
        asset_filtering_config = ASSET_FILTERING_CONFIG,
        n_threads = 5,
        resampling_freq = '1T',
        force_dw = False,
        assets = assets,
    )