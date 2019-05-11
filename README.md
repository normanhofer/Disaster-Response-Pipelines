# Disaster Response Pipelines

## Dependencies
- Machine Learning Libraries: Pandas, Sikit-Learn
- Natural Language Processing Libraries: NLTK
- Web App and Data Visualization: Flask, Plotly
- SQLlite Database Libraqries: SQLalchemy

## Execution
1. Run the following commands within the project directory:
- python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
- python train_classifier.py ../data/DisasterResponse.db classifier.pkl
