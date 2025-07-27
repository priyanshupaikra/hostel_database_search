import mysql.connector
import os

def check_duplicates():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='abcd1234',
            database='hostel_db'
        )
        
        cursor = connection.cursor()
        
        cursor.execute("""
            SELECT enrollment_number, COUNT(*) as count 
            FROM students 
            GROUP BY enrollment_number 
            HAVING COUNT(*) > 1
        """)
        
        duplicates = cursor.fetchall()
        
        if duplicates:
            print("Found duplicate enrollment numbers:")
            for enrollment, count in duplicates:
                print(f"  - {enrollment}: {count} records")
                
                cursor.execute("SELECT id, name FROM students WHERE enrollment_number = %s", (enrollment,))
                records = cursor.fetchall()
                
                print(f"    Records with enrollment {enrollment}:")
                for record_id, name in records:
                    print(f"      ID: {record_id}, Name: {name}")
                
                print("    Keeping the first record and removing duplicates...")
                
                cursor.execute("SELECT id FROM students WHERE enrollment_number = %s ORDER BY created_at ASC", (enrollment,))
                all_ids = [row[0] for row in cursor.fetchall()]
                
                ids_to_delete = all_ids[1:]
                
                for delete_id in ids_to_delete:
                    cursor.execute("SELECT photo_path FROM students WHERE id = %s", (delete_id,))
                    result = cursor.fetchone()
                    if result and os.path.exists(result[0]):
                        os.remove(result[0])
                    
                    cursor.execute("DELETE FROM students WHERE id = %s", (delete_id,))
                
                connection.commit()
                print(f"    Removed {len(ids_to_delete)} duplicate records")
        else:
            print("No duplicate enrollment numbers found!")
        
        cursor.execute("SELECT enrollment_number FROM students")
        all_enrollments = cursor.fetchall()
        print(f"\nCurrent enrollment numbers in database: {len(all_enrollments)}")
        for enrollment in all_enrollments:
            print(f"  - {enrollment[0]}")
            
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
    
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def reset_database():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='abcd1234',
            database='hostel_db'
        )
        
        cursor = connection.cursor()
        
        cursor.execute("SELECT photo_path FROM students")
        photos = cursor.fetchall()
        
        for photo in photos:
            if os.path.exists(photo[0]):
                os.remove(photo[0])
        
        cursor.execute("DELETE FROM students")
        cursor.execute("ALTER TABLE students AUTO_INCREMENT = 1")
        connection.commit()
        
        print("Database reset successfully! All student records removed.")
        
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
    
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    print("=== Hostel Database Management ===")
    print("1. Check for duplicates")
    print("2. Reset database (WARNING: This will delete all data)")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        check_duplicates()
    elif choice == "2":
        confirm = input("Are you sure you want to reset the database? Type 'YES' to confirm: ")
        if confirm == "YES":
            reset_database()
        else:
            print("Database reset cancelled.")
    else:
        print("Invalid choice!")