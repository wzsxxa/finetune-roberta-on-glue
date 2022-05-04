from transformers import pipeline
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import time
import numpy as np
import torch.nn.functional as F

def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

if __name__ == '__main__':
    # unmasker = pipeline('fill-mask', model='roberta-base')
    # results = unmasker(["The man worked as a <mask>.", "The woman worked as a <mask>."])
    # # classifier = pipeline("sentiment-analysis")
    # # results = classifier(["we are very happy", "we hope you don't hate it"])
    # #
    # for result in results:
    #     print(result)
    # raw_dataset = load_dataset("glue", "mrpc")
    # print(raw_dataset)
    # tokeniszer = AutoTokenizer.from_pretrained('roberta-base')
    # input_ids = tokeniszer('this is a sentence', 'this is another sentence')
    # print(input_ids)
    GLUE_TASKS = [ "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
    # GLUE_TASKS = ["cola"]
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir="/mnt/sevenT/wxl/transformers_cache/")
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    with open('./data', 'w') as fp:
        for i in range(len(GLUE_TASKS)):
            task = GLUE_TASKS[i]
            actual_task = "mnli" if task == "mnli-mm" else task
            dataset = load_dataset('glue', actual_task, cache_dir="/mnt/sevenT/wxl/transformers_cache/")
            metric = load_metric('glue', actual_task, cache_dir="/mnt/sevenT/wxl/transformers_cache/")
        # model_name = "distilbert-base-uncased"
            sentence1_key, sentence2_key = task_to_keys[task]
        # print(dataset)
            encoder_train_dataset = dataset['train'].map(preprocess_function, batched=True)
            encoder_val_dataset = dataset['validation'].map(preprocess_function, batched=True)
            encoder_test_dataset = dataset['test'].map(preprocess_function, batched=True)
            num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
        # print(encoder_dataset)
        # print(encoder_dataset['train'])
        # print(encoder_dataset['test'])
        # print(encoder_dataset['train'][0])
        # print(encoder_dataset['test'][0])
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir="/mnt/sevenT/wxl/transformers_cache/")
            metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
            args = TrainingArguments(
                "/mnt/sevenT/wxl/"+task,
                overwrite_output_dir=True,
                evaluation_strategy = 'epoch',
                save_strategy= 'epoch',
                do_predict=True,
                learning_rate=2e-5,
                per_device_train_batch_size= 64,
                per_device_eval_batch_size= 64,
                num_train_epochs= 5,
                weight_decay= 0.01,
                load_best_model_at_end=True,
                metric_for_best_model= metric_name
            )
            validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
            trainer = Trainer(
                model,
                args,
                train_dataset=encoder_train_dataset,
                eval_dataset=encoder_val_dataset,
                # train_dataset= encoder_dataset['train'],
                # eval_dataset= encoder_dataset[validation_key],
                tokenizer= tokenizer,
                compute_metrics= compute_metrics
            )
            bg = time.time()
            trainer.train()
            ed = time.time()
            fp.write(f"{task} "+str(ed - bg)+ "\t\t" + str(trainer.evaluate()))
            fp.write("\n")
    fp.close()
    # print(f"take {ed - bg} seconds to train")
    # print(trainer.evaluate())
    # print(trainer.predict(encoder_test_dataset))
    # print(trainer.predict(encoder_dataset['test'].map(preprocess_function, batched=True)))
