import argparse

import datetime
from loguru import logger

from scrape_utils import earnings_utils as eu

parser = argparse.ArgumentParser()

parser.add_argument(
    '-f',
    '--file',
    default='comments_stream.csv',
    type=str,
    help="path of comments output file"
)
parser.add_argument(
    '--fields',
    default=None,
    nargs='*',
    type=str
)
parser.add_argument(
    '-u',
    '--update',
    default=60,
    type=int,
    help="How often (in seconds) to query for new results"
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

while True:
    try:
        eu.stream_earnings_transcripts(args.file, args.fields, args.update)
    except:
        logger.exception('Something horrible has happened!')
