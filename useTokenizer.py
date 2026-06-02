from simpleTokenizerV1 import vocab, SimpleTokenizerV1

tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""

ids = tokenizer.encode(text)
print("Encoded : ", ids)

print("Decoded : ", tokenizer.decode(ids))