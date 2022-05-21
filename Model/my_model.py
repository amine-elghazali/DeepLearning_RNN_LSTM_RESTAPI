from pydantic import BaseModel


class Tweet(BaseModel):
    tweet_url: str
    isDisaster: bool = None
