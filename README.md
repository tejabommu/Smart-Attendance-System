# Smart-Attendance-System
**Smart Attendance System using Face Recognition**
This project implements an intelligent face recognition–based attendance system using deep learning (FaceNet + MTCNN) and MySQL for secure data storage. The system automates both registration and real-time attendance using webcam input.

**Technologies Used:**
Python
OpenCV
facenet-pytorch
MySQL
NumPy
Pickle

**Setup Instructions**
1. Install Requirements:
pip install opencv-python facenet-pytorch mysql-connector-python numpy
2. Setup MySQL
Create a MySQL server and ensure it is running.
Update your MySQL credentials in both register.py and recog.py:
python
user='root',
password='your_password',
host='127.0.0.1'

**How to Use:**
Step 1: Register a Student
python register.py
Enter student details when prompted.
Press ‘c’ to capture face and ‘q’ to quit.

Step 2: Start Attendance Recognition
python recog.py
Press ‘q’ to quit webcam.

Face is matched with stored embeddings and attendance is logged in MySQL.


**Database Tables**
students
Stores student metadata.

embeddings
Stores serialized face embeddings.

Attendance
Stores timestamped attendance entries.
