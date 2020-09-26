import cv2
from PIL import Image, ImageDraw
import numpy as np
# 调用摄像头

# 调用第一摄像头
cap = cv2.VideoCapture(0)

while True:
	# 返回
	ret, frame = cap.read()
	# BGR 是cv2保存格式，
	img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	draw = ImageDraw.Draw(img_pil)
	draw.text((100, 100), 'press q to exit', fill=(255, 255, 255))

	#
	frame = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

	cv2.imshow("widow", frame)

	if cv2.waitKey() & 0xFF == ord('q'):
		cv2.imwrite("imgs/out.jpg", frame)
		break

cap.release()

# 都去摄像头图像信息

# 图片上
