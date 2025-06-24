import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import mysql.connector
import pickle

# Initialize MTCNN and FaceNet
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval()

def initialize_database():
    # Connect to MySQL server
    conn = mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password='xxxx'
    )
    cursor = conn.cursor()

    # Create database if it does not exist
    cursor.execute("CREATE DATABASE IF NOT EXISTS face_embeddings;")
    
    # Use the newly created database
    cursor.execute("USE face_embeddings;")

    # Create students table if it does not exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            student_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255),
            course VARCHAR(255),
            department VARCHAR(255),
            section VARCHAR(50)
        )
    ''')
    
    # Create embeddings table if it does not exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            student_id VARCHAR(255) PRIMARY KEY,
            embedding LONGBLOB,
            FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    cursor.close()
    conn.close()

def save_student_info(student_id, name, course, department, section):
    # Connect to MySQL database and insert student information
    conn = mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password='xxxx',
        database='face_embeddings'
    )
    cursor = conn.cursor()
    
    # Insert student information into the students table
    cursor.execute('''
        INSERT INTO students (student_id, name, course, department, section)
        VALUES (%s, %s, %s, %s, %s)
    ''', (student_id, name, course, department, section))
    
    conn.commit()
    cursor.close()
    conn.close()

def save_embedding_to_db(student_id, embedding):
    # Serialize embedding using pickle
    embedding_blob = pickle.dumps(embedding)

    # Connect to MySQL database and insert or update embedding
    conn = mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password='xxxx',
        database='face_embeddings'
    )
    cursor = conn.cursor()
    
    # Use INSERT INTO... ON DUPLICATE KEY UPDATE to handle updates
    cursor.execute('''
        INSERT INTO embeddings (student_id, embedding)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE embedding = %s
    ''', (student_id, embedding_blob, embedding_blob))
    
    conn.commit()
    cursor.close()
    conn.close()

def capture_face(student_id):
    # Start video capture
    cap = cv2.VideoCapture(0)

    print("Press 'c' to capture your face, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Display the frame
        cv2.imshow('Webcam', frame)

        # Wait for key press
        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):  # Capture face on 'c' key press
            # Detect faces
            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    # Draw box around detected face
                    x1, y1, x2, y2 = [int(b) for b in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Get the detected face
                faces = mtcnn(frame)

                if faces is not None:
                    # Get the embedding
                    embedding = model(faces).detach().numpy()
                    # Save the embedding along with the student ID in the database
                    save_embedding_to_db(student_id, embedding)
                    print(f"Face captured and saved for Student ID: {student_id}")
                else:
                    print("No face detected.")
            else:
                print("No faces found.")

        elif key & 0xFF == ord('q'):  # Quit on 'q' key press
            print("Exiting...")
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    initialize_database()  # Ensure database is initialized

    # Gather student information
    student_id = input("Enter Student ID: ")
    name = input("Enter Name: ")
    course = input("Enter Course: ")
    department = input("Enter Department: ")
    section = input("Enter Section: ")

    # Save student information to database
    save_student_info(student_id, name, course, department, section)

    # Capture the face
    capture_face(student_id)
