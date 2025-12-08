def tone_check(grammar_output):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from datasets import load_dataset, Dataset
    import pandas as pd
    import torch
    import pandas as pd
    from datasets import load_dataset

    import os
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers.modeling_utils")
    warnings.filterwarnings("ignore")

    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)

    TRAIN = False

    new = pd.read_csv("pls.csv")
    new.head(10)

    new = new.sample(frac=1, random_state=42).reset_index(drop=True)

    min_count = new["Tone"].value_counts().min()

    df_balanced = (
    new.groupby("Tone", group_keys=False)
      .apply(lambda x: x.sample(min_count, random_state=42))
      .sample(frac=1, random_state=42)
      .reset_index(drop=True)
    )

    from datasets import load_dataset

    dataset = load_dataset("csv", data_files="pls.csv")

    Tone = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(Tone)

    def tokenize(batch):
        return tokenizer(batch["Text"], padding=True, truncation=True)

    tokenized_dataset = dataset.map(tokenize, batched=True)

    unique_labels = sorted(new["Tone"].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}

    def encode_labels(batch):
        batch["labels"] = label2id[batch["Tone"]]
        return batch

    tokenized_dataset = tokenized_dataset.map(encode_labels)

    label2id = {"formal": 0, "casual": 1, "objective": 2, "disappointed": 3}
    id2label = {v: k for k, v in label2id.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        Tone,
        num_labels=4,
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    from datasets import DatasetDict

    split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.2)

    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer
    )


    if TRAIN:
        trainer.train()
        model.save_pretrained("saved_tone_model")
        tokenizer.save_pretrained("saved_tone_model")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            "saved_tone_model",
            num_labels=4,
            id2label=id2label,
            label2id=label2id
        )
        tokenizer = AutoTokenizer.from_pretrained("saved_tone_model")


    inputs = tokenizer(["Text"], return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1)

    return id2label[int(pred)]
