from simpleTokenizerV1 import vocab, SimpleTokenizerV1
from simpleTokenizerV2 import vocabV2, SimpleTokenizerV2

# tokenizer = SimpleTokenizerV1(vocab)

# text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""

# ids = tokenizer.encode(text)
# print("Encoded : ", ids)

# print("Decoded : ", tokenizer.decode(ids))

tokenizerV2 = SimpleTokenizerV2(vocabV2)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

print("Encoded : ",tokenizerV2.encode(text))
print("Decoded : ",tokenizerV2.decode(tokenizerV2.encode(text)))

