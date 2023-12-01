import torch
from utils.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from utils.distillation_trainer import DistillationTrainer
from utils.wer import WER
from optparse import OptionParser
import numpy as np
from datasets import load_from_disk, load_dataset
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, 
                          WhisperForConditionalGeneration, 
                          WhisperProcessor, TrainerCallback)

from peft import PeftModel, PeftConfig

from huggingface_hub.hf_api import HfFolder 
HfFolder.save_token("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") # token jkot

# for default dir paths
def train(out_dir, 
          batch_size, 
          cache_dir,
          student_model_name,
          dataset_dir,
          peft_path):

    # setup data pipeline
    pipeline_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(
        pipeline_name, language="czech", 
        task="transcribe"
    )

    # setup dataset
    # dset = load_from_disk(dataset_dir)
    dset = load_dataset("jkot/merged_preprocessed_parliament_commonvoice", cache_dir=cache_dir)
    print(dset)


    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # initialize model
    student_model = WhisperForConditionalGeneration \
        .from_pretrained(student_model_name
        )
    student_model.config.forced_decoder_ids = processor \
        .get_decoder_prompt_ids(language="czech", task="transcribe")
    student_model.config.suppress_tokens = []
    if(peft_path is not None):
        # config = PeftConfig.from_pretrained(peft_path)
        student_model = PeftModel.from_pretrained(student_model, peft_path)

    print("Student model:", student_model)


    training_args = Seq2SeqTrainingArguments(
        # paths
        output_dir=out_dir,
        
        # model
        fp16=True,
        predict_with_generate=True,
        generation_max_length=225,
        
        # batch
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1 if batch_size >= 16 else 16 // batch_size,
       
        # learning rate
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=10000,
        
        # output
        metric_for_best_model="wer",
        greater_is_better=True,
        load_best_model_at_end=True,

        # feedback
        report_to=["tensorboard"],
        logging_first_step=True,        
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        save_strategy = "steps",
        evaluation_strategy="steps",
    )

    print(training_args)

    wer = WER(tokenizer=processor.tokenizer)

    # casual seq-to-seq training
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=student_model,
        data_collator=data_collator,
        compute_metrics=wer,
        tokenizer=processor.feature_extractor,
    )


    results = trainer.predict(dset["train"])
    print(results.metrics)
    with open('merged_dset_predictions_train.npy', 'wb') as f:
        np.save(f, results.predictions)
    try:
        with open('/storage/brno12-cerit/home/xvlasa15/merged_dset_predictions_train.npy', 'wb') as f:
            np.save(f, results.predictions)
    except:
        pass

    results = trainer.predict(dset["test"])
    print(results.metrics)
    with open('merged_dset_predictions_test.npy', 'wb') as f:
        np.save(f, results.predictions)
    try:
        with open('/storage/brno12-cerit/home/xvlasa15/merged_dset_predictions_test.npy', 'wb') as f:
            np.save(f, results.predictions)
    except:
        pass

if __name__ == "__main__":

    out_dir = "/storage/brno12-cerit/home/xvlasa15/"
    batch_size = 32
    cache_dir = "cache"
    student_model_name = "openai/whisper-small"
    dataset_dir = "dataset"
    peft_path = None
    train( 
        out_dir=out_dir, 
        batch_size=batch_size, 
        cache_dir=cache_dir,
        student_model_name=student_model_name,
        dataset_dir=dataset_dir,
        peft_path=peft_path
    )