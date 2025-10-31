# create a python virtualenv
python -m venv .venv
 
# activate it
 .\.venv\Scripts\Activate.ps1
 
python -m pip install --no-cache-dir -r requirements.txt
 
 uvicorn app.main:app --reload
 
 docker compose up -d
 
python -m spacy download en_core_web_lg

