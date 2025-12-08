import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration

# ----------------------------------------------------
# Load model ONCE (top-level)
# ----------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = T5TokenizerFast.from_pretrained("saved_grammar_model")
    model = T5ForConditionalGeneration.from_pretrained("saved_grammar_model")
except Exception as e:
    print("❌ Could not load saved grammar model:", e)
    raise

model.to(device)
model.eval()

# ----------------------------------------------------
# Function ONLY does inference (NO model loading!)
# ----------------------------------------------------
def grammar_check(text):
    if not text:
        return "⚠️ No input text provided."

    input_text = "grammar correction: " + text
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=64,
            num_beams=5,
            early_stopping=True
        )

    corrected = tokenizer.decode(ids[0], skip_special_tokens=True)
    return corrected
