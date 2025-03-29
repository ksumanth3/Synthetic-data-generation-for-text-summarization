from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load fine-tuned model
model = T5ForConditionalGeneration.from_pretrained("fine_tuned_summarizer")
tokenizer = T5Tokenizer.from_pretrained("fine_tuned_summarizer")

# New unseen text with task prefix
new_text = """summarize: texts = [
    "Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on enabling computers to understand, interpret, and generate human language. It plays a crucial role in applications such as chatbots, machine translation, and sentiment analysis."
]
."""


# Tokenize & generate summary
input_ids = tokenizer(new_text, return_tensors="pt", max_length=512, truncation=True).input_ids
summary_ids = model.generate(
    input_ids, 
    max_length=80,   # Keeps the summary concise  
    min_length=31,   # Ensures enough detail  
    do_sample=False,  # Keeps output deterministic  
    num_beams=5,      # More beams improve quality  
    repetition_penalty=2.0,  # Avoids unnecessary restrictions  
    no_repeat_ngram_size=2,
    length_penalty=1.2,  # Encourages shorter summaries  
    early_stopping=True  # Stops when an optimal summary is reached  
)





# Decode and print summary
print("Generated Summary:", tokenizer.decode(summary_ids[0], skip_special_tokens=True))
