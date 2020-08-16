import argparse
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

args = parser.parse_args()
du.stream_subreddit_comments(args.subreddit, args.file, args.fields)