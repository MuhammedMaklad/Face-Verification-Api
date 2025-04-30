
from app import create_app
from waitress import serve
import os
import logging

app = create_app()

if __name__ == '__main__':
    if os.environ.get('FLASK_ENV') == 'development':
        print("Server is Running in local development mode")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        serve(app, host='0.0.0.0', port=8080)