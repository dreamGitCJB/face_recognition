import face_recognition
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import datetime


class Recorder:
	pass


flag_over = 0  # 定义一个是否进行来访的标记

record_dic = {}
unknown_pic = []


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
	print('正在加载已知人员的图片')
	for dir_path, dir_names, filenames in os.walk(path):
		face_lib = []

		for filename in filenames:
			filepath = os.sep.join([dir_path, filename])
			face_image = face_recognition.load_image_file(filepath)
			face_encoding = face_recognition.face_encodings(face_image)[0]
			face_lib.append(face_encoding)
		return face_lib, filenames


face_lib, facenames = load_image("imgs")

# 调用摄像头

video_capture = cv2.VideoCapture(0)

while True:
	ret, frame = video_capture.read()
	# 通过缩小图片 提高对比效率
	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
	rgb_small_frame = small_frame[:, :, ::-1]  # 将cv2的BGR 转为RGB

	face_locations = face_recognition.face_locations(rgb_small_frame)

	face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

	face_names = []

	# 循环多张人靓
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
		font = ImageFont.truetype('SimHei.ttf', 40, encoding="unic")
		draw = ImageDraw.Draw(img_PIL)
		draw.text((left + 6, bottom - 6), name, font=font, fill=(255, 255, 255))
		frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
		save_recorder(name, frame)
	cv2.imshow("Video", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
video_capture.release()
