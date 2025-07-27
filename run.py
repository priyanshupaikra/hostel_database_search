from app import app
from database_setup import create_database
import os

if __name__ == '__main__':
    print("Setting up database...")
    create_database()
    
    print("Creating upload directory...")
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)