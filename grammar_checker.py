import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# to use gpu if available and cpu if not

try:
    tokenizer = T5TokenizerFast.from_pretrained("saved_grammar_model") # to load presaved model
    model = T5ForConditionalGeneration.from_pretrained("saved_grammar_model")
except Exception as e:
    print("Could not load saved grammar model:", e)
    raise

model.to(device)
model.eval()

def grammar_check(text):
    if not text:
        return "No input text provided." # to esnure that the output thats p[rovided is not an empty string valid

    input_text = "grammar correction: " + text  #prefix for T5 model
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask, # all the lines in this block generate the output
            max_length=64,
            num_beams=5,
            early_stopping=True
        )
    #converting ids of tokens back into human readable strings and presenting the output
    corrected = tokenizer.decode(ids[0], skip_special_tokens=True)
    return corrected
