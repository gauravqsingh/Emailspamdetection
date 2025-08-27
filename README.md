# Email / SMS Spam Detection (Text-based)

A simple, production-ready ML project for classifying messages as **Spam** or **Ham** using **TF-IDF** features and **Multinomial Naive Bayes**, with a **Streamlit** web UI.

## Project Structure
See the folder tree in this repo. Key entry points:
- Train model: `python -m src.train`
- Predict (CLI test): `python -m src.predict`
- Run app: `streamlit run app/app.py`

## Dataset
Place a file `data/spam.csv` with columns:
- `label` : `ham` or `spam`
- `message` : the text content

You can adapt other CSVs by renaming columns to these names.

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt"# emailspam-detection" 
"# Emailspamdetection" 
