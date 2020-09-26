import face_recognition
import cv2

un_know_image = face_recognition.load_image_file("imgs/face1.jpg")

know_image = face_recognition.load_image_file("imgs/face2.jpeg")

un_know_location = face_recognition.face_locations(un_know_image)

results = []

know_encodings = face_recognition.face_encodings(know_image)[0]


for i in range(len(un_know_location)):
	top, right, bottom, left = un_know_location[i]
	face_image = un_know_image[top:bottom, left:right]
	face_encoding = face_recognition.face_encodings(face_image)
	if face_encoding:
		result = {}
		matches = face_recognition.compare_faces(face_encoding, know_encodings, tolerance=0.36)
		if True in matches:
			result["face_encoding"] = face_encoding 
			result["is_view"] = True
			result["face_id"] = i + 1
			result["location"] = un_know_location[i]
			results.append(result)

			if result["is_view"]:
				print("已知面孔匹配到照片上第{}个人脸".format(result["face_id"]))

view_face_location = [i["location"] for i in results if i["is_view"]]

if(len(view_face_location)) > 0:
	for location in view_face_location:
		top, right, bottom, left = location
		start = (left, top)
		end = (right, bottom)
		cv2.rectangle(un_know_image, start, end, (0, 0, 255), thickness=2)

cv2.imshow("window", un_know_image)
cv2.waitKey()


# un_know_encodings = face_recognition.face_encodings(un_know_image)
#
#
# result = face_recognition.compare_faces(un_know_encodings, know_encodings, 0.36)
#
# print(result)
