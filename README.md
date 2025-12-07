# ✨Genie✨
Genie is an AI/ML based text analyser and corrector with grammar check, spelling check and tone check

## Key Features
- 🔤Spelling Checker: The spelling checker module processes the text to flag non-dictionary words, common typos, and phonetic misspellings. It returns structured suggestions that can be integrated into the UI. 
- 📝Grammar Checker: A smart grammar correction component that detects misplaced words, and grammar inconsistencies. It provides accurate suggestions and context-aware corrections, enabling users to produce clearer and more professional-quality writing.
- 🎯Tone Checker: The tone checker uses a trained ML model to classify text based on writing style and tone. It outputs tone predictions along with confidence scores, enabling users to understand and refine the overall mood or intent of their message.

## 🧠Tech Stack
- **Python 3.10+** – Backend logic and ML pipeline development.
- **Streamlit** – Lightweight framework for building real-time interactive applications.
- **Hugging Face Transformers (BERT/RoBERTa)** – Used for tone classification model inference.
- **PyTorch** – Neural network engine used by the Transformers model.
- **TextBlob / LanguageTool / PySpellChecker** – Modules used for grammar and spelling correction.
- **scikit-learn** – For preprocessing and potential feature engineering during model training.
- **Pandas & NumPy** – Data handling and manipulation during development.
- **NLTK (Natural Language Toolkit)** – Text preprocessing and linguistic analysis  
