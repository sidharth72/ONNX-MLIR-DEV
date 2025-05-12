import numpy as np
from transformers import GPT2Tokenizer
from PyRuntime import OMExecutionSession
import json

# 1) Load your compiled GPT-2 shared library
model_so = "models/gpt2_124M.so"
session = OMExecutionSession(shared_lib_path=model_so)

print("Input signature:", session.input_signature())
print("Output signature:", session.output_signature())

# 2) Prepare tokenizer & prompt
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
prompt = "Once upon a time, there was a"
enc = tokenizer(prompt, return_tensors="np")
input_ids = enc.input_ids.astype(np.int64)  # shape (batch, seq_len)
batch_size, seq_len = input_ids.shape

max_length = 30  # Set your desired max length

for _ in range(seq_len, max_length):
    # Prepare input for the model
    inputs = [input_ids]
    outputs = session.run(inputs)
    logits = outputs[0]  # (batch, seq_len, vocab_size)
    next_token = np.argmax(logits[:, -1, :], axis=-1)
    # Append the new token
    input_ids = np.concatenate([input_ids, next_token[:, None]], axis=1)

# Decode and print
print("Generated text:", tokenizer.decode(input_ids[0], skip_special_tokens=True))
