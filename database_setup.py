import mysql.connector
from mysql.connector import Error

def create_database():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='abcd1234'
        )
        
        cursor = connection.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS hostel_db")
        cursor.execute("USE hostel_db")
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS students (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            room_number VARCHAR(10) NOT NULL,
            enrollment_number VARCHAR(20) UNIQUE NOT NULL,
            mobile_number VARCHAR(15) NOT NULL,
            course VARCHAR(50) NOT NULL,
            photo_path VARCHAR(255) NOT NULL,
            face_encoding LONGBLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        cursor.execute(create_table_query)
        connection.commit()
        print("Database and table created successfully")
        
    except Error as e:
        print(f"Error: {e}")
    
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    create_database()