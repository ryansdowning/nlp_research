import praw

reddit_client_id = "YOUR CLIENT ID"
reddit_secret = "YOUR SECRET KEY"
reddit_agent = "YOUR AGENT NAME"
reddit = praw.Reddit(client_id=reddit_client_id, client_secret=reddit_secret, user_agent=reddit_agent)

SUBMISSION_SORTS = ("hot", "new", "rising", "controversial", "top", "gilded")
COMMENT_SORTS = ("best", "top", "new", "controversial", "old", "qa")
SUBMISSION_FIELDS = (
    "author",
    "comments",
    "clicked",
    "created_utc",
    "distinguished",
    "edited",
    "id",
    "is_original_content",
    "is_self",
    "link_flair_text",
    "locked",
    "name",
    "num_comments",
    "over_18",
    "permalink",
    "score",
    "selftext",
    "spoiler",
    "stickied",
    "subreddit",
    "title",
    "upvote_ratio",
    "url",
)
COMMENT_FIELDS = (
    "author",
    "body",
    "body_html",
    "created_utc",
    "distinguished",
    "edited",
    "id",
    "is_submitter",
    "link_id",
    "parent_id",
    "permalink",
    "replies",
    "score",
    "stickied",
    "submission",
    "subreddit",
    "subreddit_id",
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko)"
    " Chrome/58.0.3029.110 Safari/537.36",
}
