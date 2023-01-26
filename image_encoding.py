import mysql.connector
import pickle
import cv2
import os
import face_recognition

conn = mysql.connector.connect(
  host="localhost",
  user="root",
  password="root",
  database="test"
)
print(conn)
cursor = conn.cursor()

def generateImageEncodings():
    path = 'faces'
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    print(classNames)
    i=0
    cursor.execute("""delete from people""")
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]

        face_pickled_data = pickle.dumps(encode)

        cursor.execute("""insert into people(name, email, sent, encodings) values(%s, 'bsvn23@gmail.com', false, %s)""", (classNames[i], face_pickled_data))
        i+=1
    conn.commit()

def getImageEncodings():
    cursor.execute("""select encodings from people""")
    rows = cursor.fetchall()
    print("rows:", len(rows))
    image_encodings = []
    names = []
    for each in rows:
        for face_stored_pickled_data in each:
            face_data = pickle.loads(face_stored_pickled_data)
            image_encodings.append(face_data)
    cursor.execute("""select id, name from people""")
    rows = cursor.fetchall()
    for each in rows:
        names.append(each)
    return names, image_encodings