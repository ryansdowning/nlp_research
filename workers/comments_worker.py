import argparse

from loguru import logger
from rsutils import data_utils as du

parser = argparse.ArgumentParser()

parser.add_argument(
    '-s',
    '--subreddit',
    default='investing',
    type=str,
    help="subreddit to get comments from",
)
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
    '-l',
    '--log',
    action='store_true'
)

args = parser.parse_args()

if args.log:
    logger.remove()
    logger.add(
        sink='comments.log',
        level="INFO",
        format="<b><c><{time}</c></b> [{name}] <level>{level.name}</level> > {message}"
    )

try:
    du.stream_subreddit_comments(args.subreddit, args.file, args.fields)
except:
    logger.exception('Something horrible has happened!')
    raise
