from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = "gpt2-finetuned"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Ensure pad_token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Ask the user for a prompt
prompt = input("Enter your prompt: ")

inputs = tokenizer(prompt, return_tensors="pt", padding=True)

outputs = model.generate(
    **inputs,
    max_length=100,  # increased max_length for more output
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.9,
    pad_token_id=tokenizer.pad_token_id
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated Text:\n")
print(generated_text)
