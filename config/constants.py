import praw

client_id = 'ZAUZldGjWQSpUw'
secret = 'RWIDHgZ2wjdDWfrb7N30RRpjbyQ'
agent = 'data_collecting'
reddit = praw.Reddit(client_id=client_id, client_secret=secret, user_agent=agent)

SUBMISSION_SORTS = ('hot', 'new', 'rising', 'controversial', 'top', 'gilded')
COMMENT_SORTS = ('best', 'top', 'new', 'controversial', 'old', 'qa')
SUBMISSION_FIELDS = (
    'author', 'comments', 'clicked', 'created_utc', 'distinguished', 'edited', 'id',
    'is_original_content', 'is_self', 'link_flair_text', 'locked', 'name', 'num_comments',
    'over_18', 'permalink', 'score', 'selftext', 'spoiler', 'stickied', 'subreddit', 'title',
    'upvote_ratio', 'url'
)
COMMENT_FIELDS = (
    'author', 'body', 'body_html', 'created_utc', 'distinguished', 'edited', 'id', 'is_submitter',
    'link_id', 'parent_id', 'permalink', 'replies', 'score', 'stickied', 'submission', 'subreddit',
    'subreddit_id'
)