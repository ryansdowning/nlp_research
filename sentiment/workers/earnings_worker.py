import argparse
import datetime

from loguru import logger

from sentiment.config.db_config import conn
from sentiment.nlp_utils import db_utils as dbu
from sentiment.scrape_utils import earnings_utils as eu

parser = argparse.ArgumentParser()

parser.add_argument(
    '-u',
    '--update',
    default=60,
    type=int,
    help="How often (in seconds) to query for new results"
)
parser.add_argument(
    '--fields',
    default=None,
    nargs='*',
    type=str
)
parser.add_argument(
    '-f',
    '--file',
    default=None,
    type=str,
    help="path of comments output file"
)
parser.add_argument(
    '-t',
    '--table',
    default=None,
    type=str,
    help="name of table in database to write comments data to. Will use the connection parameters specified in"
         " db_config to access the database"
)
parser.add_argument(
    '-l',
    '--log',
    action='store_true'
)

args = parser.parse_args()

if args.log:
    logger.remove()
    logger.add(
        sink=f'earnings_transcripts_{datetime.date.today()}.log',
        level="INFO",
        format="<b><c><{time}</c></b> [{name}] <level>{level.name}</level> > {message}"
    )

if args.table:
    db_table = dbu.DBTable(conn, args.table)
else:
    db_table = None

while True:
    try:
        eu.stream_earnings_transcripts(args.file, args.fields, args.update)
    except:
        logger.exception('Something horrible has happened!')
