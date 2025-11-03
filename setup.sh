#!/bin/bash

# Download spaCy language model
python -m spacy download en_core_web_sm

# Download NLTK data for TextBlob
python -c "import nltk; nltk.download('brown'); nltk.download('punkt')"

echo "Setup complete!"