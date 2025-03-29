from flask import Flask, render_template, request
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)

# Load fine-tuned model
model = T5ForConditionalGeneration.from_pretrained("fine_tuned_summarizer")
tokenizer = T5Tokenizer.from_pretrained("fine_tuned_summarizer")

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    
    if request.method == "POST":
        input_text = request.form.get("text", "").strip()  # Strip whitespace to prevent empty input
        
        if input_text:
            formatted_text = "summarize: " + input_text

            # Tokenize & generate summary
            input_ids = tokenizer(formatted_text, return_tensors="pt", max_length=512, truncation=True).input_ids
            summary_ids = model.generate(
                input_ids, 
                max_length=80,    # Keeps summary concise  
                min_length=30,    # Ensures enough details  
                do_sample=False,  
                num_beams=5,      # More beams improve quality  
                repetition_penalty=1.0,  
                length_penalty=1,  
                early_stopping=True  
            )

            # Decode summary
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
