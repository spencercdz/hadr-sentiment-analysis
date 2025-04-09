# scraper.py: Twitter/X scraping functionality
from twikit import Client, TooManyRequests
from datetime import datetime
from random import randint
import time
import csv
import asyncio

# Initialize tweet and tweet limits
MINIMUM_TWEETS = 1000
tweet_count = 0

# Query (TO ADD: Custom input)
QUERY = 'myanmar earthquake'

# X login credentials
username = 'cijid91585'
email = 'cijid91585@bariswc.com'
password = '3L7p3E"<8"w+'

# Authenticate to X.com
async def login():
    await client.login(auth_info_1=username, auth_info_2=email, password=password)
    client.save_cookies('cookies.json')

async def get_tweets():
    client.load_cookies('cookies.json')
    tweets = await client.search_tweet(QUERY, product='Top')
    for tweet in tweets:
        global tweet_count
        tweet_count += 1

        tweet_data = {
            'time': tweet.created_at,
            'tweet_id': tweet.id,
            'language': tweet.lang,
            'username': tweet.user.name,
            'retweets': tweet.retweet_count,
            'favorites': tweet.favorite_count,
            'replies': tweet.reply_count,
            'text': tweet.full_text,
        }
        print(tweet_data)
        data.append(tweet_data)

    print(f'{datetime.now()} - {tweet_count} tweets found!')

if __name__ == "__main__":
    data = []

    client = Client(language='en')
    # asyncio.run(login()) # Log in and establish cookies
    asyncio.run(get_tweets()) # Scrape tweets
    # Sort tweet by timing
    #print(data)