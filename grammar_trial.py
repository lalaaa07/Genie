def grammar_check(spelling_output):
    import torch
    import evaluate
    import numpy as np
    import pandas as pd
    from torch.utils.data import TensorDataset, Dataset
    from sklearn.model_selection import train_test_split
    from transformers import T5TokenizerFast, T5ForConditionalGeneration, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers.modeling_utils")
    warnings.filterwarnings("ignore")

    TRAIN = False

    df = pd.read_csv("grammar_errors_dataset.csv")
    df.dropna(inplace=True)

    tokenizer = T5TokenizerFast.from_pretrained('t5-small')

    def preprocess_data(df, tokenizer, max_input_length=128, max_target_length=8):
        inputs = ["grammar correction: " + str(sentence) for sentence in df["incorrect"]]
        targets = [str(sentence) for sentence in df["correct"]]

        model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True, return_tensors='pt')

        labels = tokenizer(targets, max_length=max_target_length, padding="max_length", truncation=True, return_tensors='pt')

        model_inputs["labels"] = labels["input_ids"]

        model_inputs["labels"][labels["attention_mask"] == 0] = -100

        return model_inputs

    tokenized_data = preprocess_data(df, tokenizer)

    train_input_ids, val_input_ids, train_attention_mask, val_attention_mask, train_labels, val_labels = train_test_split(
        tokenized_data['input_ids'],
        tokenized_data['attention_mask'],
        tokenized_data['labels'],
        test_size=0.2,
        random_state=42
    )

    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model.to(device)

    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        predict_with_generate=True,
        report_to='none'
    )

    class CustomT5Dataset(Dataset):
        def __init__(self, input_ids, attention_mask, labels):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.labels = labels

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'labels': self.labels[idx]
            }

    train_dataset = CustomT5Dataset(
        train_input_ids,
        train_attention_mask,
        train_labels
    )

    val_dataset = CustomT5Dataset(
        val_input_ids,
        val_attention_mask,
        val_labels
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        exact_matches = [1 if pred.strip() == label.strip() else 0 for pred, label in zip(decoded_preds, decoded_labels)]
        return {"exact_match": np.mean(exact_matches)}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    if TRAIN:
        trainer.train()
        model.save_pretrained("saved_grammar_model")
        tokenizer.save_pretrained("saved_grammar_model")
    else:
        model = T5ForConditionalGeneration.from_pretrained("saved_grammar_model")
        tokenizer = T5TokenizerFast.from_pretrained("saved_grammar_model")

    model.to(device)
    model.eval()


    input_text = "grammar correction: " + spelling_output
    tokenized_input = tokenizer(input_text, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids=tokenized_input.input_ids,
        attention_mask=tokenized_input.attention_mask,
        max_length=64,
        num_beams=5,
        early_stopping=True
    )

    correct_sentence = tokenizer.decode(generated_ids[0], skip_special_tokens=True)


    return "Corrected sentence :) " + correct_sentence
