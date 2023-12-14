import numpy as np
import torch
from transformers import WhisperTokenizerFast
from datasets import load_dataset, DatasetDict


# preds_train = np.load("merged_dset_predictions_train.npy")
preds_test = np.load("merged_dset_predictions_test_trained.npy")

# print(preds_train.shape, preds_test.shape)


dataset = load_dataset("jkot/merged_preprocessed_parliament_commonvoice", cache_dir="../cache", split="test")
dataset = dataset.remove_columns(["input_features"])

tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-large-v2", language="czech", task="transcribe")

print(tokenizer.is_fast)

# dataset["train"] = dataset["train"].add_column("predictions", preds_train.tolist())
dataset= dataset.add_column("predictions", preds_test.tolist())

print(dataset[0])

# Define a function to decode the labels
def decode_labels(example):
    example['labels'] = tokenizer.batch_decode(example['labels'], skip_special_tokens=True)
    example['predictions'] = tokenizer.batch_decode(example['predictions'], skip_special_tokens=True)
    return example


# Apply the function to the 'labels' column in the dataset
dataset = dataset.map(decode_labels, batched=True, batch_size = 100, keep_in_memory = True)

#rename labels to sentence
# dataset["train"] = dataset["train"].rename_column("labels", "sentence")
dataset= dataset.rename_column("labels", "sentence")

# save dataset
dataset.save_to_disk("merged_dset_predictions_trained")
