import dlib
import cv2
from scipy.spatial import distance
from imutils import face_utils


def eye_aspect_ratio(eye):
	"""
	计算EAR值
	:param eye:
	:return:
	"""

	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	return (A + B) / (2.0 * C)


detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("libs/shape_predictor_68_face_landmarks.dat")

EAR_THRESH = 0.1

EAR_CONSEC_FRAMES = 3

#
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

# 连续侦技数
frame_counter = 5

cap = cv2.VideoCapture(1)
# 2.一定要先设置采集格式！！！
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# 3.然后再设置高清采集分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FRAME_COUNT, 30)

while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转成灰度图像
	rects = detector(gray, 1)  # 人脸检测

	if len(rects) > 0:
		shape = predictor(gray, rects[0])
		points = face_utils.shape_to_np(shape)
		left_eye = points[LEFT_EYE_START:LEFT_EYE_END + 1]
		right_eye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]

		left_ratio = eye_aspect_ratio(left_eye)
		right_ratio = eye_aspect_ratio(right_eye)

		# leftHull = cv2.convexHull(left_eye)
		# rightHull = cv2.convexHull(right_eye)
		# cv2.drawContours(frame, [leftHull], -1, (0, 255, 0), 1)
		# cv2.drawContours(frame, [rightHull], -1, (0, 255, 0), 1)

		if (left_ratio + right_ratio) / 2.0 < EAR_THRESH:
			frame_counter += 1
		else:
			if frame_counter > EAR_CONSEC_FRAMES:
				frame_counter = 0
				print("识别到活体人脸")
				# break
	cv2.imshow('11', frame)
	if cv2.waitKey(1) & 0XFF == ord('q'):
		break

cap.release()
