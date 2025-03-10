# Car-ing is Review Analytics Chatbot

## Overview
Car-ing is Sharing Chatbot is a FastAPI-based NLP application that integrates multiple transformer-based models to provide sentiment analysis, translation, question-answering, and text summarization services. The chatbot is designed for analyzing car reviews, helping users understand customer sentiments, translating feedback, extracting key information, and summarizing content effectively.

## Features
- **Sentiment Analysis:** Classifies car reviews as positive or negative using `distilbert-base-uncased-finetuned-sst-2-english`.
- **Translation (English to Spanish):** Uses `Helsinki-NLP/opus-mt-en-es` to translate customer reviews.
- **Question Answering:** Answers user queries based on car review context using `deepset/minilm-uncased-squad2`.
- **Summarization:** Generates concise summaries using `facebook/bart-large-cnn`.
- **Automated Evaluation:** Computes BLEU score for translation quality assessment.

## Tech Stack
- **Backend:** FastAPI
- **Machine Learning Models:** Hugging Face Transformers (BERT, BART, MiniLM, Opus-MT)
- **Database:** Pandas (CSV-based data handling)
- **Deployment:** Docker, Docker Compose


### Using Docker Compose
1. Start the application:
   ```bash
   docker-compose up --build
   ```
2. Stop the application:
   ```bash
   docker-compose down
   ```

## API Endpoints
| Method | Endpoint              | Description |
|--------|----------------------|-------------|
| GET    | `/health`            | Health check |
| POST   | `/analyze-sentiment` | Sentiment analysis on car reviews |
| POST   | `/translate`         | Translate text from English to Spanish |
| POST   | `/answer-question`   | Answer questions based on car reviews |
| POST   | `/summarize`         | Summarize customer reviews |
| GET    | `/run-demo`          | Run a full NLP pipeline demo |

## Example Request
```json
POST /analyze-sentiment
{
  "reviews": ["The car is amazing!", "I had a terrible experience with the service."]
}
```

## Example Response
```json
{
  "predictions": [1, 0],
  "details": [
    {"label": "POSITIVE", "score": 0.99},
    {"label": "NEGATIVE", "score": 0.98}
  ]
}
```

## Why This Project?
This project showcases expertise in:
- **NLP & AI Development:** Implementing multiple state-of-the-art NLP models for real-world applications.
- **FastAPI & Python Development:** Building scalable APIs with efficient request handling.
- **MLOps & Deployment:** Packaging ML applications in Docker for production-ready deployment.
- **Performance Evaluation:** Automating model performance evaluation (BLEU score, accuracy, F1-score).

## Future Enhancements
- Add multilingual support for translation.
- Integrate a NoSQL database for storing user interactions.
- Deploy on cloud services (AWS/GCP/Azure) with CI/CD pipelines.

## License
This project is licensed under the MIT License.


