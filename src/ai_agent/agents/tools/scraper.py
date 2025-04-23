# scraper.py: Twitter/X scraping functionality with account rotation
from twikit import Client, TooManyRequests
from datetime import datetime
from random import randint
from pathlib import Path
import json
import csv
import asyncio
import time
import re

# Define the slugify function to sanitize filenames
def slugify(text):
    """
    Convert text to a sanitized filename-friendly format
    """
    # Replace spaces and special chars with underscores
    text = re.sub(r'[^\w\s-]', '', text.lower())
    # Replace whitespace with underscores
    text = re.sub(r'[\s]+', '_', text)
    return text

class TimeoutException(Exception):
    """Custom exception for when an account hits a rate limit or timeout"""
    pass

class TwitterAccount:
    def __init__(self, account_name: str):
        """
        Initialize a Twitter account with credentials from account.json
        
        Args:
            account_name: The username of the account to use
        """
        self.account_name = account_name
        self.SCRIPT_DIR = Path(__file__).resolve().parent
        
        # Set up directory structure for accounts
        # First try the direct accounts directory in the tools directory
        self.ACCOUNTS_DIR = self.SCRIPT_DIR / 'accounts'
        if not self.ACCOUNTS_DIR.exists():
            # If not found, create the directory
            self.ACCOUNTS_DIR.mkdir(exist_ok=True)
            print(f"Created accounts directory at {self.ACCOUNTS_DIR}")
        
        self.ACCOUNTS_FILE = self.ACCOUNTS_DIR / 'accounts.json'
        self.COOKIE_FILE = self.ACCOUNTS_DIR / f'{account_name}.json'
        
        # Create a default accounts.json file if it doesn't exist
        if not self.ACCOUNTS_FILE.exists():
            default_accounts = {
                "accounts": [
                    {
                        "username": "default",
                        "email": "default@example.com",
                        "password": "defaultpassword"
                    }
                ]
            }
            with open(self.ACCOUNTS_FILE, 'w', encoding='utf-8') as f:
                import json
                json.dump(default_accounts, f, indent=4)
            print(f"Created default accounts file at {self.ACCOUNTS_FILE}")
        
        # Load account credentials
        self._load_credentials()
        
        # Initialize client
        self.client = Client(language='en')
    
    def _load_credentials(self):
        """Load account credentials from accounts.json and process them safely"""
        try:
            # First check if accounts file exists
            if not os.path.exists(self.ACCOUNTS_FILE):
                print(f"Warning: Accounts file not found at {self.ACCOUNTS_FILE}")
                print("Creating a default accounts file")
                
                # Create a basic accounts file with a default account
                default_accounts = {
                    "accounts": [
                        {
                            "username": "default",
                            "email": "default@example.com",
                            "password": "defaultpassword"
                        }
                    ]
                }
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.ACCOUNTS_FILE), exist_ok=True)
                
                # Write the default accounts file
                with open(self.ACCOUNTS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(default_accounts, f, indent=4)
                    
                print(f"Created default accounts file at {self.ACCOUNTS_FILE}")
            
            # Now load the accounts file (which should now exist)
            with open(self.ACCOUNTS_FILE, 'r', encoding='utf-8') as f:
                accounts_data = json.load(f)
                
            # Find the requested account
            account_found = False
            for account in accounts_data.get('accounts', []):
                if account.get('username') == self.account_name:
                    self.username = account.get('username', '').strip()
                    self.email = account.get('email', '').strip()
                    
                    # Handle password special characters
                    # Sometimes json loading can have issues with escape characters
                    raw_password = account.get('password', '')
                    self.password = raw_password
                    account_found = True
                    break
                    
            # If no account found but we have accounts, use the first one
            if not account_found and accounts_data.get('accounts'):
                print(f"Account '{self.account_name}' not found, using the first available account")
                first_account = accounts_data['accounts'][0]
                self.username = first_account.get('username', '').strip()
                self.email = first_account.get('email', '').strip()
                self.password = first_account.get('password', '')
                self.account_name = self.username  # Update account name to match
                self.COOKIE_FILE = self.ACCOUNTS_DIR / f'{self.account_name}.json'  # Update cookie file path
                account_found = True
            
            # If still no account found, create a default one
            if not account_found:
                print("No accounts found, creating a default account")
                self.username = "default"
                self.email = "default@example.com"
                self.password = "defaultpassword"
                
                # Add this account to the file
                if 'accounts' not in accounts_data:
                    accounts_data['accounts'] = []
                    
                accounts_data['accounts'].append({
                    "username": self.username,
                    "email": self.email,
                    "password": self.password
                })
                
                # Save the updated accounts file
                with open(self.ACCOUNTS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(accounts_data, f, indent=4)
                    
                self.account_name = self.username  # Update account name
                self.COOKIE_FILE = self.ACCOUNTS_DIR / f'{self.account_name}.json'  # Update cookie file path
        except FileNotFoundError:
            print(f"Error: Accounts file not found at {self.ACCOUNTS_FILE} and could not be created")
            # Create a default in-memory account as fallback
            self.username = "default"
            self.email = "default@example.com"
            self.password = "defaultpassword"
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON format in accounts file: {self.ACCOUNTS_FILE}")
            print(f"JSON Error: {str(e)}")
            # Create a default in-memory account as fallback
            self.username = "default"
            self.email = "default@example.com"
            self.password = "defaultpassword"
        except Exception as e:
            print(f"Error loading credentials: {str(e)}")
            # Create a default in-memory account as fallback
            self.username = "default"
            self.email = "default@example.com"
            self.password = "defaultpassword"
    
    def load_cookies(self) -> bool:
        """
        Load cookies from the account's cookie file if it exists
        
        Returns:
            True if cookies were loaded successfully, False otherwise
        """
        if self.COOKIE_FILE.exists():
            try:
                self.client.load_cookies(str(self.COOKIE_FILE))
                print(f"Cookies loaded for account: {self.account_name}")
                return True
            except Exception as e:
                print(f"Error loading cookies for {self.account_name}: {str(e)}")
                return False
        return False

    async def login(self) -> None:
        """
        Perform a fresh login using twikit
        On success, calls save_cookies()
        """
        try:
            # Print account details for debugging (without the password)
            print(f"Attempting API login for account: {self.account_name}")
            print(f"Username: {self.username}")
            print(f"Email: {self.email}")
            
            # Login with twikit
            await self.client.login(
                auth_info_1=self.username, 
                auth_info_2=self.email, 
                password=self.password
            )
            print(f"Successfully logged in as {self.account_name}")
            self.save_cookies()
        except Exception as e:
            # Print detailed error information
            print(f"Login failed for account {self.account_name}: {str(e)}")
            print("Please verify the following possible issues:")
            print("1. Account credentials are correct")
            print("2. Account is not locked or suspended")
            print("3. Twitter is not blocking login attempts from this IP")
            raise Exception(f"Login failed for account {self.account_name}")
    
    def save_cookies(self) -> None:
        """Save the current session cookies to the account's cookie file"""
        try:
            # Ensure accounts directory exists
            self.ACCOUNTS_DIR.mkdir(parents=True, exist_ok=True)
            # Save cookies
            self.client.save_cookies(str(self.COOKIE_FILE))
            print(f"Cookies saved for account: {self.account_name}")
        except Exception as e:
            print(f"Error saving cookies for {self.account_name}: {str(e)}")
    
    def get_session(self):
        """
        Return an authenticated session ready for scraping
        
        Returns:
            The authenticated twikit Client
        """
        return self.client

class TwitterScraper:
    """Twitter scraper using the TwiKit module"""
    MINIMUM_TWEETS = 2500
    # Global set to store all seen tweet IDs across all instances
    global_seen_tweet_ids = set()
    
    def __init__(self, account, query=None, product="Top"):
        self.account = account
        self.client = account.client
        self.query = query
        self.product = product  # 'Top' or 'Latest'
        
        self.data = []  # List to store the tweet data
        self.tweet_count = 0  # Counter for tweets collected
        self.seen_tweet_ids = TwitterScraper.global_seen_tweet_ids  # Use the global set for deduplication
        
        # Get the directory containing this script
        self.SCRIPT_DIR = Path(__file__).resolve().parent
        self.ACCOUNTS_DIR = self.SCRIPT_DIR / 'accounts'
        self.STATE_FILE = self.ACCOUNTS_DIR / f'{account.account_name}_state.json'
    
    def load_state(self) -> dict:
        """Load the saved pagination state"""
        state_dir = Path("data") / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a unique filename for this account and query
        account_name = self.account.account_name
        query_slug = slugify(self.query).replace("-", "_") if self.query else "default"
        state_file = state_dir / f"{account_name}_{query_slug}_{self.product.lower()}_state.json"
        
        if self.STATE_FILE.exists():
            try:
                # Use buffered I/O for better performance
                with open(self.STATE_FILE, 'r', buffering=8192) as f:
                    state = json.load(f)
                    print(f"Loaded pagination state for {self.account.account_name}")
                    return state
            except Exception as e:
                print(f"Error loading state file for {self.account.account_name}: {str(e)}")
        
        return {}
        
    def load_existing_tweet_ids(self):
        """Load existing tweet IDs from CSV to avoid duplicates"""
        if not self.query:
            return
            
        query_slug = slugify(self.query).replace("-", "_")
        output_dir = Path("data") / "raw" / query_slug
        output_path = output_dir / f"{query_slug}.csv"
        
        if output_path.exists():
            try:
                # Use a buffer size that's efficient for faster I/O
                with open(output_path, 'r', encoding='utf-8', newline='', buffering=8192) as f:
                    reader = csv.DictReader(f)
                    loaded_count = 0
                    batch_size = 10000  # Process in larger batches
                    batch = []
                    
                    for row in reader:
                        batch.append(str(row['tweet_id']))
                        loaded_count += 1
                        
                        if len(batch) >= batch_size:
                            # Add the batch to our seen tweet sets
                            self.seen_tweet_ids.update(batch)
                            TwitterScraper.global_seen_tweet_ids.update(batch)
                            batch = []
                    
                    # Process any remaining tweets in the last batch
                    if batch:
                        self.seen_tweet_ids.update(batch)
                        TwitterScraper.global_seen_tweet_ids.update(batch)
                        
                    print(f"Loaded {loaded_count} existing tweet IDs from CSV to prevent duplicates")
            except Exception as e:
                print(f"Error loading existing tweet IDs: {e}")
    
    def save_state(self, state: dict) -> None:
        """
        Save pagination state to the account's state file
        
        Args:
            state: Dictionary containing pagination state (max_position, last tweet ID, etc.)
        """
        try:
            # Ensure accounts directory exists
            self.ACCOUNTS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Create a unique filename for this account and query
            account_name = self.account.account_name
            query_slug = slugify(self.query).replace("-", "_") if self.query else "default"
            state_dir = Path("data") / "state"
            state_dir.mkdir(parents=True, exist_ok=True)
            state_file = state_dir / f"{account_name}_{query_slug}_{self.product.lower()}_state.json"
            
            with open(state_file, 'w', buffering=8192) as f:
                json.dump(state, f, indent=4)
                
            print(f"Saved pagination state for {account_name}")
        except Exception as e:
            print(f"Error saving state file for {self.account.account_name}: {str(e)}")
    
    async def scrape_until_timeout(self) -> None:
        """
        Scrape tweets until hitting a rate limit or timeout
        Saves tweets every 500 new tweets collected
        
        Raises:
            TimeoutException: When rate limit is reached or API timeout occurs
        """
        if not self.query:
            raise ValueError("Query not set. Please set a query before scraping.")
        
        # Load existing tweets from CSV first to avoid duplicates
        self.load_existing_tweet_ids()
        
        # Load the saved pagination state if it exists
        state = self.load_state()
        tweets = None
        
        # For tracking when to save to CSV
        new_tweets_since_last_save = 0
        save_threshold = 500  # Save every 500 new tweets
        
        try:
            while self.tweet_count < self.MINIMUM_TWEETS:
                try:
                    # First request or resume from saved state
                    if tweets is None:
                        if 'max_position' in state:
                            print(f"Resuming from position: {state['max_position']}")
                            # Use cursor parameter to resume from where we left off
                            tweets = await self.client.search_tweet(
                                self.query, 
                                product=self.product,
                                cursor=state.get('max_position')
                            )
                        else:
                            tweets = await self.client.search_tweet(self.query, product=self.product)
                    else:
                        # Get next page of results
                        tweets = await tweets.next()

                    if not tweets:  # If no more tweets are found
                        print(f"No more tweets found. Total collected: {self.tweet_count}")
                        break
                        
                    # Update pagination state with current cursor position
                    if hasattr(tweets, 'cursor'):
                        state = {
                            'max_position': tweets.cursor,
                            'last_tweet_id': tweets[-1].id if tweets else None,
                            'timestamp': datetime.now().isoformat()
                        }
                    
                    tweets_added_this_batch = 0
                    for tweet in tweets:
                        # Convert tweet.id to string for consistent comparison
                        tweet_id = str(tweet.id)
                        
                        # Skip if we've seen this tweet before
                        if tweet_id in self.seen_tweet_ids:
                            continue
                        
                        self.seen_tweet_ids.add(tweet_id)
                        TwitterScraper.global_seen_tweet_ids.add(tweet_id)  # Add to global set
                        self.tweet_count += 1
                        tweets_added_this_batch += 1
                        new_tweets_since_last_save += 1

                        # Clean up whitespace in tweet text - replace newlines with spaces and normalize multiple spaces
                        clean_text = tweet.full_text.replace('\n', ' ').replace('\r', ' ')
                        clean_text = ' '.join(clean_text.split())  # Normalize multiple spaces to single spaces
                        
                        tweet_data = {
                            'query': self.query.lower(),  # Always use this exact string
                            'tweet_id': str(tweet.id),  # Store as string consistently
                            'time': tweet.created_at,
                            'language': tweet.lang,
                            'username': tweet.user.name,
                            'verified': str(tweet.user.verified).capitalize(),  # Ensure 'False' not 'false' format
                            'followers': tweet.user.followers_count,
                            'location': tweet.user.location if tweet.user.location else "",  # Ensure location is never None
                            'retweets': tweet.retweet_count,
                            'favorites': tweet.favorite_count,
                            'replies': tweet.reply_count,
                            'text': clean_text
                        }

                        self.data.append(tweet_data)
                    
                    # Always sort by tweet_id to ensure chronological order
                    if tweets_added_this_batch > 0:
                        self.data.sort(key=lambda x: int(x['tweet_id']))

                    print(f'{datetime.now()} - {self.tweet_count} unique tweets collected so far (Account: {self.account.account_name})...')
                    
                    # Save to CSV every 500 new tweets
                    if new_tweets_since_last_save >= save_threshold:
                        print(f"Saving {new_tweets_since_last_save} new tweets to CSV...")
                        self.save_to_csv()
                        new_tweets_since_last_save = 0
                    
                    # Add a delay between requests to avoid rate limiting
                    delay = randint(2, 8)
                    print(f'{datetime.now()} - Waiting {delay} seconds before next request...')
                    await asyncio.sleep(delay)
                    
                except TooManyRequests:
                    print(f"Rate limit reached for account {self.account.account_name}. Saving state and switching...")
                    # Save the current pagination state before switching accounts
                    self.save_state(state)
                    # Don't lose the tweets we've collected so far
                    print(f"Collected {self.tweet_count} tweets before hitting rate limit.")
                    raise TimeoutException(f"Rate limit reached for account {self.account.account_name}")
                except Exception as e:
                    print(f"An error occurred with account {self.account.account_name}: {str(e)}")
                    # Save state on other errors too
                    self.save_state(state)
                    raise TimeoutException(f"Error with account {self.account.account_name}: {str(e)}")
                    
        except TimeoutException:
            # Propagate the timeout exception to be handled by the orchestrator
            raise
        except Exception as e:
            print(f"Unexpected error during scraping with account {self.account.account_name}: {str(e)}")
            # Try to save state even on unexpected errors
            self.save_state(state)
            raise
            
        # Sort tweets by ID (to ensure chronological order - earliest first)
        self.data.sort(key=lambda x: int(x['tweet_id']))
        print(f'{datetime.now()} - Finished collecting {self.tweet_count} unique tweets!')

    def save_to_csv(self):
        """Save the collected tweets to a CSV file"""
        if not self.data:
            print(f"{datetime.now()} - No tweets to save.")
            return
        
        # Get the path to save the CSV file
        query_slug = slugify(self.query).replace("-", "_")
        output_dir = Path("data") / "raw" / query_slug
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{query_slug}.csv"
        
        # Check if the file already exists
        file_exists = output_path.exists()
        existing_tweet_ids = set()  # To track duplicates
        all_tweets = []  # To store all tweets from CSV
        
        # Read existing CSV to get tweet IDs if file exists - optimized version
        if file_exists:
            with open(output_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                # Process in batches for better memory efficiency
                batch_size = 1000
                batch = []
                
                for row in reader:
                    # Convert tweet_id to string to ensure consistent comparisons
                    tweet_id = str(row['tweet_id'])
                    existing_tweet_ids.add(tweet_id)
                    # Also add to global set to prevent future duplicates
                    TwitterScraper.global_seen_tweet_ids.add(tweet_id)
                    batch.append(row)
                    
                    if len(batch) >= batch_size:
                        all_tweets.extend(batch)
                        batch = []
                
                # Add any remaining tweets in the last batch
                if batch:
                    all_tweets.extend(batch)
        
        # Filter out tweets that already exist in the CSV
        new_tweets = []
        for tweet in self.data:
            # Convert tweet_id to string for consistency
            tweet_id = str(tweet['tweet_id'])
            if tweet_id not in existing_tweet_ids:
                new_tweets.append(tweet)
                existing_tweet_ids.add(tweet_id)  # Add to existing set to prevent duplicates within this batch
            # Silently skip duplicates
        
        # If all tweets are duplicates, don't update the file
        if not new_tweets:
            print(f"{datetime.now()} - All tweets already exist in {output_path}. No updates needed.")
            return
            
        # Combine existing and new tweets for sorting
        if file_exists:
            # Add the new tweets to all_tweets
            all_tweets.extend(new_tweets)
        else:
            all_tweets = new_tweets
        
        # Now write all tweets to CSV file
        try:
            # Determine the field names (column headers)
            fieldnames = [
                'query', 'tweet_id', 'time', 'language', 'username', 'verified',
                'followers', 'location', 'retweets', 'favorites', 'replies', 'text'
            ]
            
            # Create a dictionary with tweet_id as key to eliminate duplicates
            unique_tweets_dict = {}
            
            for tweet in all_tweets:
                # Convert tweet_id to string for consistency
                tweet_id = str(tweet['tweet_id'])
                # Each tweet with same ID overwrites previous one, keeping only the latest version
                unique_tweets_dict[tweet_id] = tweet
            
            # Convert back to list and sort for chronological order
            unique_tweets = sorted(unique_tweets_dict.values(), 
                                  key=lambda x: int(x['tweet_id']) if isinstance(x['tweet_id'], (int, str)) else 0)
            
            # Process and write each unique tweet
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for tweet in unique_tweets:
                    # Convert tweet time to datetime if it's a string
                    if isinstance(tweet['time'], str):
                        try:
                            tweet['time'] = datetime.fromisoformat(tweet['time'])
                        except ValueError:
                            try:
                                # Try Twitter's format if ISO format fails
                                tweet['time'] = datetime.strptime(tweet['time'], '%a %b %d %H:%M:%S +0000 %Y')
                            except ValueError:
                                pass
                    
                    # Format the datetime to a readable string
                    if isinstance(tweet['time'], datetime):
                        tweet['time'] = tweet['time'].strftime('%Y-%m-%d_%H-%M-%S')
                    
                    # Clean up whitespace in tweet text if it's a string
                    if 'text' in tweet and isinstance(tweet['text'], str):
                        # Replace newlines with spaces and normalize multiple spaces
                        clean_text = tweet['text'].replace('\n', ' ').replace('\r', ' ')
                        tweet['text'] = ' '.join(clean_text.split())  # Normalize multiple spaces
                    
                    # Ensure query is always lowercase but preserve the original search term
                    if 'query' in tweet and tweet['query'] is not None:
                        tweet['query'] = tweet['query'].lower()
                    
                    # Ensure verified is always 'True' or 'False' (capitalized)
                    if 'verified' in tweet:
                        tweet['verified'] = str(tweet['verified']).capitalize()
                    
                    # Ensure location is never None
                    if 'location' in tweet and tweet['location'] is None:
                        tweet['location'] = ""
                    
                    writer.writerow(tweet)
        
            message = "appended to" if file_exists else "created new"
            print(f"{datetime.now()} - Successfully wrote {len(unique_tweets)} tweets to {output_path} ({len(new_tweets)} new)")
        except Exception as e:
            print(f"Error saving to CSV: {e}")

async def load_accounts():
    """
    Load all available accounts from accounts.json
    
    Returns:
        List of account names
    """
    script_dir = Path(__file__).resolve().parent
    accounts_file = script_dir / 'accounts' / 'accounts.json'
    
    try:
        with open(accounts_file, 'r') as f:
            accounts_data = json.load(f)
        
        return [account['username'] for account in accounts_data['accounts']]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading accounts: {str(e)}")
        return []

async def main():
    """Main function to run the scraper with a single account and switch between Top/Latest when needed"""
    # Get the search query
    query = str(input("Enter a search query (e.g. 'myanmar earthquake'): "))
    if not query:
        print("No query provided. Exiting.")
        return
        
    # Get minimum number of tweets to collect (default is 2500)
    try:
        min_tweets_input = input("Enter minimum number of tweets to collect (default: 2500): ").strip()
        min_tweets = int(min_tweets_input) if min_tweets_input else 2500
    except ValueError:
        print("Invalid number. Using default of 2500 tweets.")
        min_tweets = 2500
    
    # Ask user which search type to start with
    search_type_input = input("Start with (T)op tweets or (L)atest tweets? [T/R] Default: T: ").strip().upper()
    initial_product = "Latest" if search_type_input == "L" else "Top"
    print(f"Searching for: {query} (starting with {initial_product} tweets)")
    
    # Load the first available account
    account_names = await load_accounts()
    if not account_names:
        print("No accounts found in accounts.json")
        return
    
    account_name = account_names[0]  # Just use the first account
    print(f"Using account: {account_name}")
    
    # Initialize account
    account = TwitterAccount(account_name)
    
    # Try to load cookies, if fail then login
    if not account.load_cookies():
        print(f"Need to login with account {account_name}")
        await account.login()
    
    # Collection of tweets from both search types
    all_tweets = []
    tried_search_types = []
    current_product = initial_product
    
    # Try both search types if needed
    while len(tried_search_types) < 2:  # We have 2 modes: Top and Latest
        if current_product not in tried_search_types:
            tried_search_types.append(current_product)
        
        print(f"\n=== Trying {current_product} tweets ===")
        
        try:
            # Initialize scraper for this search mode
            scraper = TwitterScraper(account, query=query, product=current_product)
            
            # Set custom minimum tweet count if specified
            scraper.MINIMUM_TWEETS = min_tweets
            
            # Start scraping until timeout/rate-limit
            await scraper.scrape_until_timeout()
            
            # Add collected tweets to our overall collection
            if scraper.data:
                print(f"Adding {len(scraper.data)} tweets from {current_product} search to collection.")
                all_tweets.extend(scraper.data)
                
            print(f"Total tweets collected so far: {len(all_tweets)}")
            
            # If we've collected enough tweets, we can stop
            if len(all_tweets) >= scraper.MINIMUM_TWEETS:
                print(f"Collected required {scraper.MINIMUM_TWEETS} tweets. Stopping.")
                break
        except Exception as e:
            print(f"Error searching for {current_product} tweets: {str(e)}")
        
        # Switch to the other search mode
        if current_product == "Top":
            current_product = "Latest"
        else:
            current_product = "Top"
            
        # If we've tried both search types, exit the loop
        if current_product in tried_search_types:
            break
            
        # Add a small delay before switching search types
        delay = randint(5, 15)
        print(f"Waiting {delay} seconds before switching to {current_product} tweets search...")
        await asyncio.sleep(delay)
            
    # Save all collected tweets
    if all_tweets:
        print(f"Saving {len(all_tweets)} tweets collected from {', '.join(tried_search_types)} searches")
        # Create a temporary scraper just for saving tweets
        temp_scraper = TwitterScraper(account, query=query, product=current_product)
        temp_scraper.MINIMUM_TWEETS = min_tweets
        # Make sure we've loaded existing tweet IDs before saving
        temp_scraper.load_existing_tweet_ids()
        temp_scraper.data = all_tweets
        temp_scraper.save_to_csv()
    else:
        print("No tweets collected.")
        
    print("Scraping complete.")

if __name__ == "__main__":
    asyncio.run(main())