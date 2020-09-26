import face_recognition

image = face_recognition.load_image_file("imgs/陈金宝.jpg")

face_image = face_recognition.load_image_file("imgs/陈金宝.jpg")

known_face_encodings = face_recognition.face_encodings(image)

compare_face_encoding = face_recognition.face_encodings(face_image)[0]

matches = face_recognition.compare_faces(known_face_encodings, compare_face_encoding, tolerance=0.39)

print(matches)


