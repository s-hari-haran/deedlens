web: uvicorn api.app:app --host 0.0.0.0 --port $PORT
worker: celery -A worker.celery_app worker --loglevel=info --pool=solo
streamlit: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
