import cv2
import numpy as np
import mysql.connector
import torch
import pickle  # Add this import statement
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1

# Initialize MTCNN and FaceNet
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval()

# Set to keep track of recognized employees
recognized_employees = set()

def load_embeddings_from_db():
    conn = mysql.connector.connect(
        host='127.0.0.1',  # Change if your MySQL server is on a different host
        user='root',  # Your MySQL username
        password='xxxx',  # Your MySQL password
        database='face_embeddings'  # Your MySQL database name
    )
    cursor = conn.cursor()

    # Query to fetch embeddings
    cursor.execute('SELECT student_id, embedding FROM embeddings')
    results = cursor.fetchall()
    
    employee_ids = []
    embeddings_list = []

    for student_id, embedding_blob in results:
        # Deserialize the embedding
        embedding = pickle.loads(embedding_blob)
        employee_ids.append(student_id)
        embeddings_list.append(embedding)

    cursor.close()
    conn.close()
    
    # Convert embeddings list to a NumPy array
    return employee_ids, np.vstack(embeddings_list)

def initialize_attendance_table():
    conn = mysql.connector.connect(
        host='127.0.0.1',  # Change if your MySQL server is on a different host
        user='root',  # Your MySQL username
        password='xxxx',  # Your MySQL password
        database='face_embeddings'  # Your MySQL database name
    )
    cursor = conn.cursor()

    # Drop the Attendance table if it exists
    cursor.execute('DROP TABLE IF EXISTS Attendance')

    # Create Attendance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Attendance (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            employee_id VARCHAR(255),
            time_of_attendance DATETIME
        )
    ''')
    conn.commit()
    cursor.close()
    conn.close()

def recognize_faces(embeddings_list, employee_ids):  # Reset for each session
    cap = cv2.VideoCapture(0)
    print("Starting face recognition... Press 'q' to quit.")


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Detect faces
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                # Draw box around detected face
                x1, y1, x2, y2 = [int(b) for b in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Get the detected faces
            faces = mtcnn(frame)
            if faces is not None:
                # Get the embeddings for the detected faces
                face_embeddings = model(faces).detach().numpy()

                # Compare with stored embeddings
                for embedding in face_embeddings:
                    # Calculate distances between the detected face and stored embeddings
                    distances = np.linalg.norm(embeddings_list - embedding, axis=1)
                    min_distance_index = np.argmin(distances)
                    min_distance = distances[min_distance_index]

                    # Define a threshold for face recognition
                    threshold = 0.8  # Adjust this value based on your tests

                    if min_distance < threshold:
                        employee_id = employee_ids[min_distance_index]

                        # Record attendance only if not already recognized
                        if employee_id not in recognized_employees:
                            record_attendance(employee_id)
                            recognized_employees.add(employee_id)  # Mark this employee as recognized

        # Display the frame
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

def record_attendance(employee_id):
    time_now = datetime.now()
    print(f"Attempting to record attendance for Employee ID: {employee_id} at {time_now}")

    # Connect to the database and insert attendance record
    try:
        conn = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password='xxxx',
            database='face_embeddings'
        )
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO Attendance (employee_id, time_of_attendance)
            VALUES (%s, %s)
        ''', (employee_id, time_now))
        
        conn.commit()
        print(f"Attendance recorded for Employee ID: {employee_id} at {time_now}")
    except mysql.connector.Error as err:
        print(f"Error recording attendance: {err}")
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    initialize_attendance_table()  # Ensure Attendance table is initialized
    employee_ids, embeddings_list = load_embeddings_from_db()  # Load embeddings from MySQL
    recognize_faces(embeddings_list, employee_ids)
