# KNN WHISPER

This projects implements training, evaluation and dataset processing ZPJa project. 

## Requirements
Project uses huggingface and pytorch. You can install all of the requirements using
```
pip install -r requirements.txt
```
## Get ASR predictions
To get predictions from ASR system, use the src/predict.py script

## Process the predictions for language model training
To save the predictions as huggingface dataset for LM training use the src/process_predictions.py and src/process_parliament_predictions.py scripts

## Train the language model to correct errors
Use src/train_corrector.py

## Evaluate the language model
Use src/eval.py

## MetaCentrum
All the computation was done at the MetaCentrum.