import re
import emoji as ej

class TextCleaner:
    def __init__(self, remove_profanity=False, preserve_quotes=True):
        self.remove_profanity = remove_profanity
        self.preserve_quotes = preserve_quotes

        self.abbreviations = {
            r'\bimo\b': 'in my opinion',
            r'\brt\b': 'retweet',
            r'\bu\b': 'you',
            r'\bur\b': 'your',
            r'\bomg\b': 'oh my god',
            r'\blmao\b': 'laughing my ass off',
            r'\bidk\b': 'i do not know',
            r'\bsmh\b': 'shaking my head',
            r'\basap\b': 'as soon as possible',
            r'\bdm\b': 'direct message',
            r'\bbtw\b': 'by the way',
            r'\btbh\b': 'to be honest',
            r'\bfyi\b': 'for your information'
        }

        self.contractions = {
            r"\bcan't\b": "cannot",
            r"\bwon't\b": "will not",
            r"\bi'm\b": "i am",
            r"\bthey're\b": "they are",
            r"\bwe're\b": "we are",
            r"\byou're\b": "you are",
            r"\bit's\b": "it is",
            r"\bdon't\b": "do not",
            r"\bdoesn't\b": "does not",
            r"\bdidn't\b": "did not",
            r"\bhasn't\b": "has not",
            r"\bhaven't\b": "have not",
            r"\bhadn't\b": "had not",
            r"\bwouldn't\b": "would not",
            r"\bcouldn't\b": "could not",
            r"\bshouldn't\b": "should not",
            r"\bi've\b": "i have",
            r"\byou've\b": "you have",
            r"\bthey've\b": "they have",
            r"\bwho's\b": "who is",
            r"\bwhat's\b": "what is",
            r"\bthat's\b": "that is",
            r"\blet's\b": "let us",
            r"\bi'll\b": "i will",
            r"\byou'll\b": "you will",
            r"\bthey'll\b": "they will",
            r"\bit'll\b": "it will",
            r"\bthere's\b": "there is"
        }

        self.profanity_list = [r'\bshit\b', r'\bfuck\b', r'\bass\b', r'\bdamn\b']  # Extend as needed

    def encode_hashtags(self, text):
        return re.sub(r'#(\w+)', lambda m: f" hashtag_{m.group(1)} ", text)

    def encode_mentions(self, text):
        return re.sub(r'@(\w+)', lambda m: f" mention_{m.group(1)} ", text)

    def encode_emojis(self, text):
        demojized = ej.demojize(text)
        return re.sub(r':([^:]+):', lambda m: f" emoji_{m.group(1)} ", demojized)

    def expand_abbreviations(self, text):
        for pattern, replacement in self.abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def expand_contractions(self, text):
        for pattern, replacement in self.contractions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def remove_noise(self, text):
        text = re.sub(r'\b(?:https?://|www\.)\S+\b', '', text)
        text = re.sub(r'\S+@\S+', '', text)  # Remove emails
        return text

    def remove_profanity_words(self, text):
        for word in self.profanity_list:
            text = re.sub(word, '[censored]', text, flags=re.IGNORECASE)
        return text

    def clean(self, text):
        if not isinstance(text, str):
            return ''

        text = self.encode_emojis(text)
        text = self.encode_hashtags(text)
        text = self.encode_mentions(text)
        text = self.expand_abbreviations(text)
        text = self.expand_contractions(text)
        text = self.remove_noise(text)

        # Base pattern
        if self.preserve_quotes:
            pattern = r"[^a-zA-Z0-9\s.,!?_'’\"]"  # Preserve quotes
        else:
            pattern = r"[^a-zA-Z0-9\s.,!?_'’]"    # Remove quotes

        text = re.sub(pattern, '', text)

        # Space out punctuation
        if self.preserve_quotes:
            text = re.sub(r'([.,!?_"\'])', r' \1 ', text)
        else:
            text = re.sub(r'([.,!?_])', r' \1 ', text)

        if self.remove_profanity:
            text = self.remove_profanity_words(text)

        text = text.lower()
        text = ' '.join(text.split())

        return text