import subprocess
import sys
import os

def install_requirements():
    print("Installing required packages...")
    
    packages = [
        'Flask==2.3.3',
        'mysql-connector-python==8.1.0',
        'opencv-python==4.8.0.76',
        'tensorflow==2.13.0',
        'face-recognition==1.3.0',
        'numpy==1.24.3',
        'Pillow==10.0.0',
        'Werkzeug==2.3.7',
        'dlib==19.24.2',
        'cmake==3.27.2'
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")

def create_directories():
    print("Creating necessary directories...")
    
    directories = [
        'static',
        'static/uploads',
        'templates',
        'models'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory already exists: {directory}")

def setup_mysql():
    print("\nMySQL Setup Instructions:")
    print("1. Install MySQL Server on your system")
    print("2. Create a database user with the following credentials:")
    print("   - Host: localhost")
    print("   - Username: root")
    print("   - Password: password")
    print("3. Make sure MySQL service is running")
    print("4. The application will automatically create the 'hostel_db' database")

def main():
    print("=== Hostel Face Recognition System Setup ===\n")
    
    create_directories()
    print()
    
    install_requirements()
    print()
    
    setup_mysql()
    print()
    
    print("=== Setup Complete ===")
    print("To run the application:")
    print("1. Make sure MySQL is running")
    print("2. Run: python run.py")
    print("3. Open browser and go to: http://localhost:5000")

if __name__ == "__main__":
    main()