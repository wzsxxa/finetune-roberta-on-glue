from transformers import pipeline
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import torch.nn.functional as F

def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
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
    dataset = load_dataset('glue', 'cola')
    metric = load_metric('glue', 'cola')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    task_to_key = {
        "cola": ("sentence", None)
    }
    sentence1_key, sentence2_key = task_to_key["cola"]
    print(dataset)
    encoder_dataset = dataset.map(preprocess_function, batched=True)
    print(encoder_dataset)
    print(encoder_dataset['train'][0])
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    args = TrainingArguments(
        "./test-glue",
        evaluation_strategy = 'epoch',
        save_strategy= 'epoch',
        learning_rate=2e-5,
        per_device_train_batch_size= 64,
        per_device_eval_batch_size= 64,
        num_train_epochs= 5,
        weight_decay= 0.01,
        load_best_model_at_end=True,
        metric_for_best_model= "metthews_correlation"
    )
    validation_key = "validation"
    trainer = Trainer(
        model,
        args,
        train_dataset= encoder_dataset['train'],
        eval_dataset= encoder_dataset[validation_key],
        tokenizer= tokenizer,
        compute_metrics= compute_metrics
    )
    print("xian")
