import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        # store the vocab, token to id mapping
        self.str_to_int = vocab 
        
        # id to token mapping
        self.int_to_str = {
            i: s for s,i in vocab.items()
        }
    
    def encode(self, text):
        # Tokenize
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        
        # For now we'll ignore whitespaces
        preprocessed = [
            item.strip() for item in preprocessed if item.strip() 
            ]
        
        # Get IDs for tokens
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids    

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        