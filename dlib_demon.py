
import dlib
from skimage import io

detector = dlib.get_frontal_face_detector()

win = dlib.image_window()


img = io.imread('imgs/陈金宝.jpg')

dets = detector(img, 1)

win.set_image(img)
win.add_overlay(dets)

dlib.hit_enter_to_continue()


