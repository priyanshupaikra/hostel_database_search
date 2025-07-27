# Hostel Face Recognition System

A comprehensive full-stack web application for hostel student management using advanced face recognition technology powered by deep learning CNN models.

## Features

- **Student Registration**: Register students with personal details and face recognition
- **Multi-Search Options**: Search by name, room number, enrollment number, or photo
- **Deep CNN Face Recognition**: Advanced ResNet50-based face recognition with triplet loss
- **Admin Panel**: Secure admin interface for managing student records
- **Real-time Processing**: Instant face detection and matching
- **Responsive Design**: Modern UI with Tailwind CSS

## Technology Stack

- **Backend**: Python, Flask
- **Database**: MySQL
- **Frontend**: HTML, CSS, Tailwind CSS, JavaScript
- **Machine Learning**: TensorFlow, OpenCV, face_recognition library, Custom CNN
- **Image Processing**: OpenCV, PIL

## File Structure

```
hostel-face-recognition/
├── app.py                    # Main Flask application
├── run.py                    # Application runner
├── database_setup.py         # Database configuration
├── face_model.py            # Face recognition utilities
├── deep_cnn_model.py        # Advanced CNN model
├── config.py                # Configuration settings
├── install.py               # Installation script
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── static/
│   └── uploads/            # Uploaded images storage
├── templates/
│   ├── index.html          # Home page
│   ├── register.html       # Student registration
│   ├── search.html         # Student search
│   ├── admin.html          # Admin panel
│   └── edit_student.html   # Edit student form
└── models/                 # Trained ML models storage
```

## Installation

### Prerequisites

- Python 3.8+
- MySQL Server
- Git

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hostel-face-recognition
   ```

2. **Run installation script**
   ```bash
   python install.py
   ```

3. **Setup MySQL Database**
   - Install MySQL Server
   - Create user with credentials:
     - Host: localhost
     - Username: root
     - Password: password
   - Start MySQL service

4. **Run the application**
   ```bash
   python run.py
   ```

5. **Access the application**
   - Open browser and navigate to: `http://localhost:5000`

## Usage

### Student Registration

1. Navigate to the registration page
2. Fill in student details:
   - Full Name
   - Room Number
   - Enrollment Number (unique)
   - Mobile Number
   - Course (B.Tech, M.Tech, BCA, MCA, B.Sc, M.Sc, MBA, PhD)
   - Upload a clear face photo
3. Submit the form

### Student Search

1. Go to the search page
2. Choose search method:
   - **By Name**: Enter student name
   - **By Room Number**: Enter room number
   - **By Enrollment Number**: Enter enrollment number
   - **By Photo**: Upload a photo for face recognition
3. View search results

### Admin Panel

1. Access admin panel from home page
2. Enter admin password: `admin123`
3. View all registered students
4. Edit or delete student records

## Deep Learning Model

The system uses an advanced CNN architecture:

- **Base Model**: ResNet50 pre-trained on ImageNet
- **Custom Layers**: Dense layers with batch normalization and dropout
- **Loss Function**: Triplet loss for face embedding learning
- **Features**: 512-dimensional face embeddings
- **Similarity Metric**: Cosine similarity and Euclidean distance

## Database Schema

### Students Table

```sql
CREATE TABLE students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    room_number VARCHAR(10) NOT NULL,
    enrollment_number VARCHAR(20) UNIQUE NOT NULL,
    mobile_number VARCHAR(15) NOT NULL,
    course VARCHAR(50) NOT NULL,
    photo_path VARCHAR(255) NOT NULL,
    face_encoding LONGBLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Configuration

Edit `config.py` to customize:

- Database credentials
- File upload settings
- Face recognition thresholds
- Admin password
- Model paths

## Security Features

- Secure file upload with type validation
- SQL injection protection with parameterized queries
- Admin authentication
- Unique constraint on enrollment numbers
- Input sanitization

## Performance Optimization

- Efficient face encoding storage using pickle
- Optimized CNN inference
- Batch processing capabilities
- Image preprocessing pipeline
- Database indexing on frequently queried fields

## Troubleshooting

### Common Issues

1. **MySQL Connection Error**
   - Ensure MySQL server is running
   - Verify credentials in config.py
   - Check if port 3306 is available

2. **Face Detection Failed**
   - Ensure uploaded image has a clear, visible face
   - Check image format (JPG, PNG supported)
   - Verify proper lighting in the photo

3. **Package Installation Error**
   - Install CMAKE: `pip install cmake`
   - Install dlib: `pip install dlib`
   - For Windows: Install Visual Studio Build Tools

4. **Model Loading Error**
   - Check if TensorFlow is properly installed
   - Verify model file path
   - Ensure sufficient system memory

## API Endpoints

- `GET /` - Home page
- `GET,POST /register` - Student registration
- `GET,POST /search` - Student search
- `GET,POST /admin` - Admin panel
- `GET /delete_student/<id>` - Delete student
- `GET,POST /edit_student/<id>` - Edit student

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Add tests
5. Submit pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
- Create an issue on GitHub
- Contact the development team
- Check documentation and troubleshooting guide

## Future Enhancements

- Multi-face detection in single image
- Real-time video recognition
- Mobile application
- Advanced analytics dashboard
- Backup and restore functionality
- Email notifications
- Attendance tracking integration

## Example Screenshort
![main_page](1..png)
