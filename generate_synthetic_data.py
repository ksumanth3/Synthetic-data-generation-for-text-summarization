import pandas as pd
from transformers import pipeline

# Load the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Sample texts
texts = [
    "Artificial Intelligence (AI) is rapidly transforming various industries by automating repetitive tasks, improving efficiency, and enabling data-driven decision-making. Businesses are increasingly integrating AI-driven solutions to optimize workflows and reduce operational costs.",
    "Many companies today rely on AI-powered tools to enhance their decision-making processes. By analyzing large volumes of data, AI models help businesses identify trends, make accurate predictions, and develop more effective strategies for market growth.",
    "The healthcare industry has embraced AI to improve patient care and medical diagnostics. AI-driven algorithms can analyze medical images, detect diseases early, and assist doctors in providing accurate treatment plans, ultimately leading to better healthcare outcomes.",
    "Self-driving cars, a major innovation in the automotive industry, rely on AI and machine learning to navigate roads safely. By processing real-time sensor data, these vehicles can make split-second driving decisions, reducing the risk of accidents and enhancing transportation efficiency.",
    "AI-powered chatbots are revolutionizing customer service by providing instant responses and handling a wide range of inquiries. These chatbots use natural language processing (NLP) to understand user queries and deliver accurate information, reducing the workload on human agents.",
    "The education sector is leveraging AI to create personalized learning experiences for students. AI-based platforms analyze student progress, adapt teaching methods, and recommend customized study materials to enhance learning outcomes and engagement.",
    "In manufacturing, robotic automation powered by AI is reducing human labor costs while increasing productivity. Automated assembly lines, quality control systems, and predictive maintenance algorithms help manufacturers streamline operations and minimize downtime.",
    "Fraud detection in banking has become more effective with AI-driven models. These models analyze transaction patterns in real time, identify anomalies, and detect fraudulent activities, helping financial institutions safeguard customer assets.",
    "Voice assistants such as Siri, Alexa, and Google Assistant use AI-powered natural language processing to understand and respond to user commands. These virtual assistants perform tasks, answer questions, and provide a seamless hands-free experience for users.",
    "Predictive analytics is helping businesses forecast customer preferences by analyzing past behaviors and market trends. Companies leverage these insights to tailor marketing campaigns, improve product offerings, and enhance customer satisfaction."
]


# Generate synthetic summaries
data = []

for text in texts:
    summary = summarizer(
    text,
    max_length=80,  # Allow for a more detailed summary
    min_length=30,  # Ensure meaningful content
    do_sample=True, 
    temperature=0.7,
    top_k=50,
    top_p=0.9
)[0]["summary_text"]


    print(f"Original: {text}")
    print(f"Summary: {summary}\n")


    data.append({"text": text, "summary": summary})

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("synthetic_summarization_data.csv", index=False)
print("Synthetic data saved to synthetic_summarization_data.csv")
