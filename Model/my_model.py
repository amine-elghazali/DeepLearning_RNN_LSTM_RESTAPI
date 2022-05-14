from pydantic import BaseModel


class Tweet(BaseModel):
    tweetMsg: str
    isDisaster: bool = None
