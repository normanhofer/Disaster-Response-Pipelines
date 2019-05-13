# Disaster Response Pipelines

This project aims to categorize disaster related text messages so that aid organizations can read out the needs of a large amount of text messages and use their resources in a targeted way.
The project is based on real messages that were sent during disaster events, provided by figure eight. The analysis consists of 3 parts:

1. ETL pipeline for processing text data.
2. Machine Learning pipeline that builds and trains a classifier.
3. Interacitve web app that allows to type in a message. The app also displays the classification results in real time.

## Dependencies
- Python 3
- Machine Learning Libraries: Pandas, Sikit-Learn
- Natural Language Processing Libraries: NLTK
- Web App and Data Visualization: Flask
- SQLlite Database Libraqries: SQLalchemy
- Using regular expressions: re
- Modify Python runtime environment: sys

## Installation

Clone this repository:


## Execution
1. Run the following commands within the project directory:
- python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
- python train_classifier.py ../data/DisasterResponse.db classifier.pkl
2. Start the app via running the following command in the directory of the app: python run.py 
3. Visit the following url: http://0.0.0.0:3001/
