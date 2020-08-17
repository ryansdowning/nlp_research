import argparse
import logging
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
    logger = logging.getLogger('rsutils')
    sh = logging.StreamHandler()
    fh = logging.FileHandler('comments.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s (%(levelname)s): %(message)s')

    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)

    logger.setLevel(logging.INFO)

try:
    du.stream_subreddit_comments(args.subreddit, args.file, args.fields)
except:
    logging.exception('Something horrible has happened!')
    raise