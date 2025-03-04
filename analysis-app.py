import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline, logging
import evaluate
from sklearn.metrics import accuracy_score, f1_score

# Suppress unnecessary warnings and download NLTK data
logging.set_verbosity(logging.ERROR)
nltk.download('punkt', quiet=True)

# Task 1: Sentiment Analysis
# Load and preprocess the dataset
df = pd.read_csv('data/car_reviews.csv', delimiter=';', quotechar='"', encoding='utf-8-sig')
df.columns = ['review', 'sentiment']
df['sentiment'] = df['sentiment'].str.strip().str.lower().map({'positive': 1, 'negative': 0})

# Initialize sentiment pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Process reviews and get predictions
predicted_labels = sentiment_analyzer(df['review'].tolist(), truncation=True, batch_size=4)
predictions = [1 if res['label'] == 'POSITIVE' else 0 for res in predicted_labels]

# Calculate metrics
accuracy_result = accuracy_score(df['sentiment'], predictions)
f1_result = f1_score(df['sentiment'], predictions)

# Task 2: Translation
# Extract first two sentences from the first review
first_review = df['review'].iloc[0]
sentences = sent_tokenize(first_review)
first_two_sentences = ' '.join(sentences[:2]) if len(sentences) >= 2 else first_review

# Initialize translation pipeline
translator = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")

# Translate text
translation_result = translator(first_two_sentences, truncation=True, max_length=512)
translated_review = translation_result[0]['translation_text']

# Calculate BLEU score with proper reference format
with open('data/reference_translations.txt', 'r', encoding='utf-8') as f:
    reference_translations = [line.strip() for line in f]

bleu = evaluate.load("bleu")
bleu_score = bleu.compute(
    predictions=[translated_review],
    references=[[ref] for ref in reference_translations]  # Fix: Properly format references as list of lists
)['bleu']  # Fix: Extract the BLEU score value from the result dictionary

# Task 3: Question Answering
context = df['review'].iloc[1]
question = "What did he like about the brand?"

# Initialize QA pipeline
qa_model = pipeline("question-answering", model="deepset/minilm-uncased-squad2")

# Get answer
qa_result = qa_model(question=question, context=context, truncation=True)
answer = qa_result['answer'] if qa_result['score'] > 0.01 else "No relevant answer found"

# Task 4: Summarization
last_review = df['review'].iloc[-1]

# Initialize summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Calculate appropriate summary length
word_count = len(last_review.split())
max_len = min(55, max(40, int(word_count * 0.3)))
min_len = int(max_len * 0.7)

# Generate summary
summary_result = summarizer(last_review, max_length=max_len, min_length=min_len, truncation=True)
summarized_text = summary_result[0]['summary_text']

# Print results
print("=== Final Results ===")
print(f"1. Sentiment Analysis\n   Accuracy: {accuracy_result:.4f}\n   F1 Score: {f1_result:.4f}")
print(f"\n2. Translation\n   BLEU Score: {bleu_score:.4f}")  # Fix: Format BLEU score properly
print(f"\n3. QA Answer\n   '{answer}'")
print(f"\n4. Summarization\n   '{summarized_text}'")
