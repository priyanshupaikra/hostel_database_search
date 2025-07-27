from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import mysql.connector
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import face_recognition
from datetime import datetime
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='abcd1234',
        database='hostel_db'
    )

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_face(image_path):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    if face_encodings:
        return face_encodings[0]
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        room_number = request.form['room_number']
        enrollment_number = request.form['enrollment_number']
        mobile_number = request.form['mobile_number']
        course = request.form['course']
        
        if 'photo' not in request.files:
            flash('No photo uploaded')
            return redirect(request.url)
        
        file = request.files['photo']
        if file.filename == '':
            flash('No photo selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{enrollment_number}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            face_encoding = encode_face(filepath)
            if face_encoding is None:
                flash('No face detected in the image')
                os.remove(filepath)
                return redirect(request.url)
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM students WHERE enrollment_number = %s", (enrollment_number,))
            existing_student = cursor.fetchone()
            
            if existing_student:
                flash('Error: Student with this enrollment number already exists')
                os.remove(filepath)
                cursor.close()
                conn.close()
                return redirect(request.url)
            
            try:
                cursor.execute("""
                    INSERT INTO students (name, room_number, enrollment_number, mobile_number, course, photo_path, face_encoding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (name, room_number, enrollment_number, mobile_number, course, filepath, pickle.dumps(face_encoding)))
                
                conn.commit()
                flash('Student registered successfully!')
                return redirect(url_for('register'))
                
            except mysql.connector.Error as err:
                flash(f'Database error: {str(err)}')
                os.remove(filepath)
                
            finally:
                cursor.close()
                conn.close()
    
    return render_template('register.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        search_type = request.form.get('search_type')
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if search_type == 'photo':
            if 'search_photo' not in request.files:
                flash('No photo uploaded')
                return redirect(request.url)
            
            file = request.files['search_photo']
            if file and allowed_file(file.filename):
                filename = secure_filename(f"search_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                search_encoding = encode_face(filepath)
                os.remove(filepath)
                
                if search_encoding is None:
                    flash('No face detected in the uploaded image')
                    return redirect(request.url)
                
                cursor.execute("SELECT * FROM students")
                students = cursor.fetchall()
                
                best_match = None
                best_distance = float('inf')
                
                for student in students:
                    stored_encoding = pickle.loads(student[7])
                    distance = face_recognition.face_distance([stored_encoding], search_encoding)[0]
                    
                    if distance < 0.6 and distance < best_distance:
                        best_distance = distance
                        best_match = student
                
                if best_match:
                    result = {
                        'id': best_match[0],
                        'name': best_match[1],
                        'room_number': best_match[2],
                        'enrollment_number': best_match[3],
                        'mobile_number': best_match[4],
                        'course': best_match[5],
                        'photo_path': best_match[6]
                    }
                    cursor.close()
                    conn.close()
                    return render_template('search.html', results=[result])
                else:
                    flash('No matching student found')
        
        else:
            search_value = request.form.get('search_value')
            if search_type == 'name':
                cursor.execute("SELECT * FROM students WHERE name LIKE %s", (f"%{search_value}%",))
            elif search_type == 'room_number':
                cursor.execute("SELECT * FROM students WHERE room_number = %s", (search_value,))
            elif search_type == 'enrollment_number':
                cursor.execute("SELECT * FROM students WHERE enrollment_number = %s", (search_value,))
            
            students = cursor.fetchall()
            results = []
            for student in students:
                results.append({
                    'id': student[0],
                    'name': student[1],
                    'room_number': student[2],
                    'enrollment_number': student[3],
                    'mobile_number': student[4],
                    'course': student[5],
                    'photo_path': student[6]
                })
            
            cursor.close()
            conn.close()
            return render_template('search.html', results=results)
        
        cursor.close()
        conn.close()
    
    return render_template('search.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        password = request.form.get('password')
        if password == 'abcd1234':
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM students")
            students = cursor.fetchall()
            
            results = []
            for student in students:
                results.append({
                    'id': student[0],
                    'name': student[1],
                    'room_number': student[2],
                    'enrollment_number': student[3],
                    'mobile_number': student[4],
                    'course': student[5],
                    'photo_path': student[6]
                })
            
            cursor.close()
            conn.close()
            return render_template('admin.html', students=results, authenticated=True)
        else:
            flash('Invalid password')
    
    return render_template('admin.html', authenticated=False)

@app.route('/delete_student/<int:student_id>')
def delete_student(student_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT photo_path FROM students WHERE id = %s", (student_id,))
    result = cursor.fetchone()
    
    if result:
        photo_path = result[0]
        if os.path.exists(photo_path):
            os.remove(photo_path)
        
        cursor.execute("DELETE FROM students WHERE id = %s", (student_id,))
        conn.commit()
        flash('Student deleted successfully')
    
    cursor.close()
    conn.close()
    return redirect(url_for('admin'))

@app.route('/edit_student/<int:student_id>', methods=['GET', 'POST'])
def edit_student(student_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if request.method == 'POST':
        name = request.form['name']
        room_number = request.form['room_number']
        mobile_number = request.form['mobile_number']
        course = request.form['course']
        
        cursor.execute("""
            UPDATE students SET name = %s, room_number = %s, mobile_number = %s, course = %s
            WHERE id = %s
        """, (name, room_number, mobile_number, course, student_id))
        
        conn.commit()
        flash('Student updated successfully')
        cursor.close()
        conn.close()
        return redirect(url_for('admin'))
    
    cursor.execute("SELECT * FROM students WHERE id = %s", (student_id,))
    student = cursor.fetchone()
    
    if student:
        student_data = {
            'id': student[0],
            'name': student[1],
            'room_number': student[2],
            'enrollment_number': student[3],
            'mobile_number': student[4],
            'course': student[5],
            'photo_path': student[6]
        }
        cursor.close()
        conn.close()
        return render_template('edit_student.html', student=student_data)
    
    cursor.close()
    conn.close()
    return redirect(url_for('admin'))

if __name__ == '__main__':
    app.run(debug=True)