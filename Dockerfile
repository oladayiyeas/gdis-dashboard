from python:3.11-slim
WORKDIR /app 
COPY ./app 
RUN pip install --no-cache-dir -r requirements.txt 
EXPOSE 7860 
CMD ["panel", "serve","dashboard_app.py", "--address","0.0.0.0","--port","7860","--allow-websocket-origin=*"]
