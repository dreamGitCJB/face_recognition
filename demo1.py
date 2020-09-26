import face_recognition
from PIL import Image, ImageDraw
import cv2

#  这个方法主要用于加载要识别的人脸图像， 加载放回的数据Numpy 数组， 记录这个图片的所有的像素的特征向量
image = face_recognition.load_image_file('imgs/陈金宝.jpg')

'''定位图中的所有的人脸的像素位置
 返回值是一个列表形式， 列表中每一行是一张人脸的位置信息（top, right, bottom, left）主要用于标注图片上的人脸'''
locations = face_recognition.face_locations(image)

# for face_location in  locations:
# 	top, right, bottom, left = face_location;
# 	start = (left, top)
# 	end = (right, bottom)
# 	cv2.rectangle(image, start, end, (0, 0, 255), thickness=2)
#
# cv2.imshow('window', image)
# cv2.waitKey()
	# face_image = image[top:bottom, left:right]
	# pil_image = Image.fromarray(face_image)
	# pil_image.show()

''' face_landmarks 识别人脸的关键特征点
	参数仍然是待检测的图像对象， 返回值是包含面部特征点字典的列表， 列表的长度就是图像中的人脸数
	面部特征包括一下几个部分： nose_bridge（鼻梁）、 right_eyebrow（右眼眉）、left_eyebrow(左眼眉)、right_eye、left_eye、chin（下巴）、nose_tip（下鼻部）、bottom_lip(下嘴唇)
	勾勒出人脸轮廓 '''

# face_landmarks = face_recognition.face_landmarks(image)
#
# pil_image = Image.fromarray(image)
#
# d = ImageDraw.Draw(pil_image)
#
# for face_landmark in face_landmarks:
# 	facial_features = [
# 		'chin',
# 		'left_eyebrow',
# 		'right_eyebrow',
# 		'nose_bridge',
# 		'nose_tip',
# 		'left_eye',
# 		'right_eye',
# 		'bottom_lip'
# 	]
# 	for facial_feature in facial_features:
# 		print("{}每个人的面部特征显示在以下位置：{}".format(facial_feature, face_landmark[facial_feature]))
# 		d.line(face_landmark[facial_feature], width=5) #  直接调用PIL中line 方法
#
# pil_image.show()


''' face_encodings 获取图像文件中的所有面部编码信息
 	返回值是一个编码列表， 参数仍然是要识别的图像对象  如果后续梵文是， 需要注意加上索引或者遍历来进行反问， 每张人脸的编码信息学是一个128 维的向量
 	面部编码信息是进行人像对比的重要参数。'''

face_encodings = face_recognition.face_encodings(image)

for face_encoding in  face_encodings:
	print("信息编码长度为:{} \n 编码信息为：{}".format(len(face_encoding), face_encoding))

''' 由面不比阿妈信息进行面部识别匹配
 	1。 主要用于匹配两个面部特征编码，利用这两个特征向量的内积来衡量相似度， 根据阈值来确认是否是同一个人。
 	2、第一个参数就是给出一个面部编码列表（很多张脸 len = 1或> 1）， 第二个参数就是给出单个面部编码（一张脸）,
 		compare_faces 会将第二个参数值中的编码信息与第一个参数中的所有编码信息依次匹配，返回值是一个boolean列表， 匹配成功则True，匹配失败则返回失败， 顺序与第一个参数中的脸部编码顺序一致
 	3、参数里又一个tolerance=0.6 , 大家可以根据实际的效果进行调整， 一般经验值是0.39 tolerance越小，匹配越严格
 	'''

# print(face_landmarks)


