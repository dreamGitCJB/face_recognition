import dlib
import face_recognition
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import datetime
from scipy.spatial import distance
from imutils import face_utils
import time


class Recorder:
	pass


flag_over = 0  # 定义一个是否进行来访的标记

record_dic = {}  # 记录的字典
unknown_pic = []  # 未知图片


def save_recorder(name, frame):
	global record_dic
	global flag_over
	global unknown_pic
	if flag_over == 1:
		return

	try:
		record = record_dic[name]
		second_diff = (datetime.datetime.now() - record.times[-1]).total_seconds()

		if second_diff < 60 * 10:
			return
		record.times.append(datetime.datetime.now())
		print('更新记录', record_dic, record.times)
	except KeyError:
		newRec = Recorder()
		newRec.times = [datetime.datetime.now()]
		record_dic[name] = newRec
		print('新增记录', record_dic, newRec.times)
		return
	if name == '未知图像':
		s = str(record_dic[name].times[-1])
		filename = s[:10] + s[-6:] + '.jpg'
		cv2.imwrite(facenames, frame)
		unknown_pic.append(filename)
		return


def load_image(path):
	# '正在加载已知人员的图片'
	for dir_path, dir_names, filenames in os.walk(path):
		face_lib = []
		for filename in filenames:
			filepath = os.sep.join([dir_path, filename])
			face_image = face_recognition.load_image_file(filepath)
			face_encoding = face_recognition.face_encodings(face_image)[0]
			face_lib.append(face_encoding)
		return face_lib, filenames


face_lib, facenames = load_image("imgs")


def eye_aspect_ratio(eye):
	"""
	计算EAR值
	:param eye:
	:return:
	"""

	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	print((A + B) / (2.0 * C))
	return (A + B) / (2.0 * C)


detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("libs/shape_predictor_68_face_landmarks.dat")

EAR_THRESH = 0.2

EAR_CONSEC_FRAMES = 3

#
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

# 连续侦技数
frame_counter = 5

# 调用摄像头

video_capture = cv2.VideoCapture(0)

IS_SLEEP = 0


# 人脸对比
def face_compare(frame):
	# 通过缩小图片 提高对比效率
	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
	rgb_small_frame = small_frame[:, :, ::-1]  # 将cv2的BGR 转为RGB

	face_locations = face_recognition.face_locations(rgb_small_frame)

	face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

	face_names = []

	# 循环多张人脸
	for face_encoding in face_encodings:
		matches = face_recognition.compare_faces(face_lib, face_encoding, tolerance=0.39)

		name = '未知头像'

		if True in matches:
			first_matches_index = matches.index(True)
			name = facenames[first_matches_index][:-4]  # 取出文件上对应的人名
		face_names.append(name)
	for (top, right, bottom, left), name in zip(face_locations, face_names):
		top *= 4
		right *= 4
		bottom *= 4
		left *= 4

		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

		img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		font = ImageFont.truetype('SimHei.ttf', 40, encoding="unic")  # 自己安装SimHei字体 mac自带的字体不兼容汉字
		draw = ImageDraw.Draw(img_PIL)
		draw.text((left + 6, bottom - 6), name, font=font, fill=(255, 255, 255))
		frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
		save_recorder(name, frame)

# 2.一定要先设置采集格式！！！
# video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# 3.然后再设置高清采集分辨率
# video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:

	# if (IS_SLEEP == 1):
	# 	time.sleep(10)
	# 	IS_SLEEP = 0
	ret, frame = video_capture.read()

	# 活体检测
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
				print('-----------------')
				frame_counter = 0
				# 调用人体比较
				face_compare(frame)
				IS_SLEEP = 1

	cv2.imshow("Video", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video_capture.release()
