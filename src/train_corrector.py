from collections import namedtuple
from utils.wer import WER
from optparse import OptionParser

from datasets import load_from_disk
from transformers import (AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, 
                          AutoTokenizer, TrainerCallback)

model_checkpoint = "facebook/mbart-large-50"

print("Using model: " + model_checkpoint)

#load dataset from disk
dataset = load_from_disk("../common_voice_predictions_text")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, legacy=False)
wer = WER(tokenizer=tokenizer)


if "mbart" in model_checkpoint:
    tokenizer.src_lang = "cs_CZ"
    tokenizer.tgt_lang = "cs_CZ"

max_input_length = 128
max_target_length = 128

def preprocess_function(examples):
    inputs = [ex for ex in examples["predictions"]]
    targets = [ex for ex in examples["sentence"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)


model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

batch_size = 16
model_name = model_checkpoint.split("/")[-1]

# save model to /storage/brno12-cerit/home/xvlasa15/model_name
args = Seq2SeqTrainingArguments(
    f"/storage/brno12-cerit/home/xvlasa15/{model_name}",
    evaluation_strategy = "epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
    report_to=["tensorboard"],

)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)




trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=wer
)

# class PrintExamplesCallback(TrainerCallback):
#     def on_epoch_end(self, args, state, control, **kwargs):
#         if state.is_local_process_zero:
#             print("epoch: ", state.epoch)
#             for i in range(10):
#                 prediction = trainer.predict(tokenized_datasets["test"][i])
#                 decoded_prediction = tokenizer.decode(prediction, skip_special_tokens=True)
#                 print(f"Prediction {i}: {decoded_prediction}")
#                 decoded_label = tokenizer.decode(tokenized_datasets["test"][i]["labels"], skip_special_tokens=True)
#                 print(f"Label {i}: {decoded_label}")

# trainer.add_callback(PrintExamplesCallback)

trainer.train()
trainer.model.save_pretrained(args.output_dir)      
tokenizer.save_pretrained(args.output_dir)