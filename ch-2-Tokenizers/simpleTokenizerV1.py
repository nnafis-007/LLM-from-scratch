import re

# Build vocab
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character in Corpus :", len(raw_text))
# print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_tokens = sorted(set(preprocessed)) # keep only unique words
# all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_tokens)
print("Vocab size: ", vocab_size)

vocab = {token:integer for integer,token in enumerate(all_tokens)}
print("Vocab Sample :")
for i, item in enumerate(vocab.items()):
    print(item)
    if i > 10:
        break



print("="*20)
class SimpleTokenizerV1:
    def __init__(self, vocab):
        # store the vocab, token to id mapping
        self.str_to_int = vocab 
        
        # id to token mapping
        self.int_to_str = {
            i: s for s,i in vocab.items()
        }
    
    def encode(self, text):
        # Tokenize -> Split by punctuation, -- or whitespace (\s)
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text) 
        
        # For now we'll ignore whitespaces
        preprocessed = [
            item.strip() for item in preprocessed if item.strip() 
            ]
        
        # Get IDs for tokens
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids    

    def decode(self, ids):
        # get strings from ids
        text = " ".join([self.int_to_str[i] for i in ids])

        # punctuation also joined by \s, so search for pattern where punc trailed by whitespaces
        # then replace then pattern by only the punctuation (r'\1')
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) 
        return text
        