# scraper.py: Twitter/X scraping functionality
from twikit import Client, TooManyRequests
from datetime import datetime
from random import randint
from pathlib import Path
import csv
import asyncio

class TwitterScraper:
    def __init__(self, query=None):
        # Initialize tweet and tweet limits
        self.MINIMUM_TWEETS = 2500
        self.tweet_count = 0
        self.seen_tweet_ids = set()
        self.data = []
        
        # Get the directory containing this script
        self.SCRIPT_DIR = Path(__file__).resolve().parent
        self.COOKIES_PATH = self.SCRIPT_DIR / 'cookies.json'
        
        # X login credentials
        self.username = 'cijid91585'
        self.email = 'cijid91585@bariswc.com'
        self.password = '3L7p3E"<8"w+'
        
        # Initialize client
        self.client = Client(language='en')
        
        # Set query if provided
        self.query = query

    async def login(self):
        """Authenticate to X.com"""
        await self.client.login(auth_info_1=self.username, auth_info_2=self.email, password=self.password)
        self.client.save_cookies(str(self.COOKIES_PATH))
        print(f"Cookies saved to: {self.COOKIES_PATH}")

    async def get_tweets(self):
        """Scrape tweets based on query"""
        if not self.query:
            raise ValueError("Query not set. Please set a query before scraping.")

        self.client.load_cookies(str(self.COOKIES_PATH))
        
        while self.tweet_count < self.MINIMUM_TWEETS:
            try:
                if self.tweet_count == 0:
                    tweets = await self.client.search_tweet(self.query, product='Top')
                else:
                    tweets = await tweets.next()

                if not tweets:  # If no more tweets are found
                    print(f"No more tweets found. Total collected: {self.tweet_count}")
                    break
                    
                for tweet in tweets:
                    # Skip if we've seen this tweet before
                    if tweet.id in self.seen_tweet_ids:
                        continue
                    
                    self.seen_tweet_ids.add(tweet.id)
                    self.tweet_count += 1

                    tweet_data = {
                        'query': self.query,
                        'tweet_id': tweet.id,
                        'time': tweet.created_at,
                        'language': tweet.lang,
                        'username': tweet.user.name,
                        'verified': tweet.user.verified,
                        'followers': tweet.user.followers_count,
                        'location': tweet.user.location,
                        'retweets': tweet.retweet_count,
                        'favorites': tweet.favorite_count,
                        'replies': tweet.reply_count,
                        'text': tweet.full_text,
                    }

                    self.data.append(tweet_data)

                print(f'{datetime.now()} - {self.tweet_count} unique tweets collected so far...')
                
                # Add a delay between requests to avoid rate limiting
                delay = randint(2, 8)
                print(f'{datetime.now()} - Waiting {delay} seconds before next request...')
                await asyncio.sleep(delay)
                
            except TooManyRequests:
                print("Rate limit reached. Exiting search now...")
                break
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                break

        # Sort tweets by ID
        self.data.sort(key=lambda x: x['tweet_id'])
        print(f'{datetime.now()} - Finished collecting {self.tweet_count} unique tweets!')

    def save_to_csv(self):
        """Save scraped tweets to CSV"""
        if not self.data:
            print("No data to save!")
            return

        # Clean query string for folder and file naming
        sanitized_query = self.query.replace(' ', '_').replace('/', '_').replace('\\', '_').lower()
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        data_file = f'{sanitized_query}_{current_time}.csv'

        # Locate paths
        project_root = self.SCRIPT_DIR.parent.parent.parent
        data_dir = project_root / 'data'
        raw_data_dir = data_dir / 'raw'
        query_dir = raw_data_dir / sanitized_query

        # Create directories if they don't exist
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        query_dir.mkdir(parents=True, exist_ok=True)
        
        # Full path for the output file
        output_path = query_dir / data_file

        print(f"Saving {len(self.data)} tweets to: {output_path}")

        try:
            # Define the field names in a specific order
            fieldnames = [
                'query',
                'tweet_id',
                'time',
                'language',
                'username',
                'verified',
                'followers',
                'location',
                'retweets',
                'favorites',
                'replies',
                'text'
            ]

            # Save data to CSV
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for tweet in self.data:
                    # Convert tweet time to datetime if it's a string
                    if isinstance(tweet['time'], str):
                        try:
                            # Parse the Twitter timestamp format
                            tweet_time = datetime.strptime(tweet['time'], "%a %b %d %H:%M:%S +0000 %Y")
                        except ValueError:
                            # If parsing fails, try to use the time as is
                            tweet_time = tweet['time']
                    else:
                        tweet_time = tweet['time']

                    # Ensure all fields are properly formatted
                    row = {
                        'query': self.query,
                        'tweet_id': str(tweet['tweet_id']),
                        'time': tweet_time.strftime("%Y-%m-%d_%H-%M-%S") if isinstance(tweet_time, datetime) else str(tweet_time),
                        'language': tweet['language'],
                        'username': tweet['username'],
                        'verified': str(tweet['verified']).lower(),
                        'followers': str(tweet['followers']),
                        'location': tweet['location'] if tweet['location'] else '',
                        'retweets': str(tweet['retweets']),
                        'favorites': str(tweet['favorites']),
                        'replies': str(tweet['replies']),
                        'text': tweet['text'].replace('\n', ' ').strip()
                    }
                    writer.writerow(row)
        
            print(f'{datetime.now()} - Successfully saved {len(self.data)} tweets to {output_path}')
            
        except Exception as e:
            print(f"Error saving CSV file: {str(e)}")
            raise

async def main():
    """Main function to run the scraper"""
    while True:
        try:
            query = str(input("Enter a search query (e.g. 'myanmar earthquake'): "))
            if query:
                print(f"Searching for {query}...")
                break
            else:
                print("Query cannot be empty. Please try again.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again.")

    # Initialize scraper with query
    scraper = TwitterScraper(query)
    
    # Scrape tweets
    await scraper.get_tweets()
    
    # Save data to CSV
    try:
        scraper.save_to_csv()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Data was not saved to CSV. Exiting now...")
        exit()

if __name__ == "__main__":
    asyncio.run(main())