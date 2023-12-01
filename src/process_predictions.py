import numpy as np
import torch
from transformers import WhisperProcessor
from datasets import load_dataset, DatasetDict
preds_train = np.load("common_voice_predictions_train.npy")
preds_test = np.load("common_voice_predictions.npy")

dataset = DatasetDict()

dataset["train"] = load_dataset("mozilla-foundation/common_voice_13_0","cs", split="train+validation")
dataset["test"] = load_dataset("mozilla-foundation/common_voice_13_0","cs", split="test")
dataset = dataset.remove_columns(
            ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes", "audio"])


processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", language="czech", task="transcribe")

preds_train_text = []
for pred in preds_train:
    pred_text =  processor.tokenizer.decode(pred, skip_special_tokens=True)
    preds_train_text.append(pred_text)

preds_test_text = []
for pred in preds_test:
    pred_text =  processor.tokenizer.decode(pred, skip_special_tokens=True)
    preds_test_text.append(pred_text)


# add columns for predictions
dataset["train"] = dataset["train"].add_column("predictions", preds_train_text)
dataset["test"] = dataset["test"].add_column("predictions", preds_test_text)

# save dataset
dataset.save_to_disk("common_voice_predictions")