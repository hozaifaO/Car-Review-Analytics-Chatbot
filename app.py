from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import os
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline, logging
import evaluate
from sklearn.metrics import accuracy_score, f1_score
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Car-ing is sharing Chatbot", version="1.0")

# Suppress unnecessary warnings and download NLTK data
logging.set_verbosity(logging.ERROR)
nltk.download('punkt', quiet=True)

# Initialize models
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
translator = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
qa_model = pipeline("question-answering", model="deepset/minilm-uncased-squad2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load BLEU evaluator
bleu = evaluate.load("bleu")

# Request and response models
class SentimentRequest(BaseModel):
    reviews: List[str]

class TranslationRequest(BaseModel):
    text: str

class QARequest(BaseModel):
    question: str
    context: str

class SummarizationRequest(BaseModel):
    text: str
    max_length: int = 55
    min_length: int = 40

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Sentiment analysis endpoint
@app.post("/analyze-sentiment")
async def analyze_sentiment(request: SentimentRequest):
    try:
        predicted_labels = sentiment_analyzer(request.reviews, truncation=True, batch_size=4)
        predictions = [1 if res['label'] == 'POSITIVE' else 0 for res in predicted_labels]
        return {"predictions": predictions, "details": predicted_labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Translation endpoint
@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        translation_result = translator(request.text, truncation=True, max_length=512)
        translated_text = translation_result[0]['translation_text']
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Question answering endpoint
@app.post("/answer-question")
async def answer_question(request: QARequest):
    try:
        qa_result = qa_model(question=request.question, context=request.context, truncation=True)
        answer = qa_result['answer'] if qa_result['score'] > 0.01 else "No relevant answer found"
        return {"answer": answer, "confidence": qa_result['score']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Summarization endpoint
@app.post("/summarize")
async def summarize(request: SummarizationRequest):
    try:
        summary_result = summarizer(
            request.text, 
            max_length=request.max_length, 
            min_length=request.min_length, 
            truncation=True
        )
        summarized_text = summary_result[0]['summary_text']
        return {"summary": summarized_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Full demo endpoint that runs the complete example
@app.get("/run-demo")
async def run_demo():
    try:
        # Load data
        data_path = os.getenv("DATA_PATH", "data")
        df = pd.read_csv(f'{data_path}/car_reviews.csv', delimiter=';', quotechar='"', encoding='utf-8-sig')
        df.columns = ['review', 'sentiment']
        df['sentiment'] = df['sentiment'].str.strip().str.lower().map({'positive': 1, 'negative': 0})
        
        # Task 1: Sentiment Analysis
        predicted_labels = sentiment_analyzer(df['review'].tolist(), truncation=True, batch_size=4)
        predictions = [1 if res['label'] == 'POSITIVE' else 0 for res in predicted_labels]
        accuracy_result = accuracy_score(df['sentiment'], predictions)
        f1_result = f1_score(df['sentiment'], predictions)
        
        # Task 2: Translation
        first_review = df['review'].iloc[0]
        sentences = sent_tokenize(first_review)
        first_two_sentences = ' '.join(sentences[:2]) if len(sentences) >= 2 else first_review
        translation_result = translator(first_two_sentences, truncation=True, max_length=512)
        translated_review = translation_result[0]['translation_text']
        
        # BLEU score
        with open(f'{data_path}/reference_translations.txt', 'r', encoding='utf-8') as f:
            reference_translations = [line.strip() for line in f]
        
        bleu_score = bleu.compute(
            predictions=[translated_review],
            references=[[ref] for ref in reference_translations]
        )['bleu']
        
        # Task 3: Question Answering
        context = df['review'].iloc[1]
        question = "What did he like about the brand?"
        qa_result = qa_model(question=question, context=context, truncation=True)
        answer = qa_result['answer'] if qa_result['score'] > 0.01 else "No relevant answer found"
        
        # Task 4: Summarization
        last_review = df['review'].iloc[-1]
        word_count = len(last_review.split())
        max_len = min(55, max(40, int(word_count * 0.3)))
        min_len = int(max_len * 0.7)
        summary_result = summarizer(last_review, max_length=max_len, min_length=min_len, truncation=True)
        summarized_text = summary_result[0]['summary_text']
        
        return {
            "sentiment_analysis": {
                "accuracy": float(accuracy_result),
                "f1_score": float(f1_result)
            },
            "translation": {
                "original": first_two_sentences,
                "translated": translated_review,
                "bleu_score": float(bleu_score)
            },
            "question_answering": {
                "question": question,
                "context_excerpt": context[:100] + "...",
                "answer": answer
            },
            "summarization": {
                "original_length": word_count,
                "summary": summarized_text
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)