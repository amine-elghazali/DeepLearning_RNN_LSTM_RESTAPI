from pydantic import BaseModel

from enums.disaster_degrees import disasterDegree


class Tweet(BaseModel):
    tweet_url: str
    isDisaster: disasterDegree = None
