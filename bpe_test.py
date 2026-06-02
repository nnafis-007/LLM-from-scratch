from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

# BPE can break down unknown words
ids = tokenizer.encode("Akwirw ier")
print("ids : ", ids)

text = tokenizer.decode(ids)
print("decoded : ", text)