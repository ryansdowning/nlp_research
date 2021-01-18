import argparse

import prawcore
from loguru import logger

from sentiment.config.db_config import conn
from sentiment.nlp_utils import db_utils as dbu
from sentiment.scrape_utils import reddit_utils as ru

parser = argparse.ArgumentParser()

parser.add_argument(
    '-s',
    '--subreddit',
    default='investing',
    type=str,
    help="subreddit to get comments from",
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
        sink=f'submissions_{args.subreddit}.log',
        level="INFO",
        format="<b><c><{time}</c></b> [{name}] <level>{level.name}</level> > {message}"
    )

if args.table:
    db_table = dbu.DBTable(conn, args.table)
else:
    db_table = None

while True:
    try:
        ru.stream_subreddit_submissions(
            subreddit=args.subreddit, data_fields=args.fields, file=args.file, table=db_table
        )
    except prawcore.exceptions.ServerError:
        logger.debug('prawcore.exceptions.ServerError encountered, this could be caused by a server'
                     ' overload. Restarting submissions stream')
    except:
        logger.exception('Something horrible has happened!')
