import cv2
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation

face_position = []


def detect_final(img2):
    """
        This function checks face in final image before returning
        it to main file
        :param img2: final image
        :return: 1, if human face is present in image
                 0, otherwise
    """
    flag = 0
    global face_position
    image = img2.copy()

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)
    face = []
    for c in faces:
        if (c[2] * c[3]) > 14000:
            face.append(c)

    if len(face) == 1:
        flag = 1
    return flag


def blur_image(ima, contour_points):
    """
        This function smooths the edges of human in image
        :param ima: image with changed background
        :param contour_points : edges contours where blurring is to be done
        :return: image with smooth edges
    """
    image = ima.copy()
    blurred_img = cv2.GaussianBlur(image, (21, 21), 0)
    mask = np.zeros(image.shape, np.uint8)

    cv2.drawContours(mask, contour_points, -1, (255, 255, 255), 5)
    output = np.where(mask == np.array([255, 255, 255]), blurred_img, image)
    return output


def contour_detect(im):
    """
        This Function find boundary contours of human in image.
        These contours point will later be to smooth edges of human in image.
        :param im: image with changed background
        :return: contour points of boundary of face
    """
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    roi_contour = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        height = im.shape[1]
        if h < (height * 0.4):
            continue
        roi_contour.append(contour)
    return roi_contour


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
        :param image: image with changed background
        :param width: width of required image
        :param height: height of required image
        :param inter: interpolation method
        :return: image with reduced size
    """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def reduce_size(image, parameter):
    """
        Function to change size of image
        :param image: image with changed background
        :param parameter: list containing size, padding, background-color information
        :return: resized image
    """
    size = (parameter[0], parameter[1])
    resized_image = image_resize(image, size[0], size[1])
    return resized_image


def remove(image, parameter):
    """
        Function to replace background with solid color background

        :param image: input image
        :param parameter: list containing size, padding, background-color information
        :return: image with background changed to required background color
    """
    global face_position
    segmentor = SelfiSegmentation()
    white = (parameter[6], parameter[7], parameter[8])
    imgNoBg = segmentor.removeBG(image, white, threshold=.7)

    # print(imgNoBg.shape)
    face_position[0] = face_position[0] - (int(0.3 * face_position[2])) - parameter[4]
    face_position[1] = face_position[1] - (int(0.4 * face_position[3])) - parameter[2]
    face_position[2] = face_position[2] + (int(0.6 * face_position[2])) + parameter[5]
    face_position[3] = face_position[3] + (int(0.8 * face_position[3])) + parameter[3]

    parameter[3] = 0
    if face_position[0] < 0:
        parameter[4] = 0 - face_position[0]
        face_position[0] = 0
    if face_position[1] < 0:
        parameter[2] = 0 - face_position[1]
        face_position[1] = 0
    if face_position[2] > imgNoBg.shape[1]:
        parameter[5] = face_position[2] - imgNoBg.shape[1]
        face_position[2] = imgNoBg.shape[1]
    if face_position[3] > imgNoBg.shape[0]:
        parameter[3] = 0
        face_position[3] = imgNoBg.shape[0]

    # print(*face_position)
    imgNoBg = imgNoBg[face_position[1]:face_position[1] + face_position[3],
              face_position[0]:face_position[0] + face_position[2]]
    imgNoBg = cv2.copyMakeBorder(imgNoBg, parameter[2], parameter[3], parameter[4], parameter[5], cv2.BORDER_CONSTANT,
                                 value=(parameter[6], parameter[7], parameter[8]))

    imgNoBg = reduce_size(imgNoBg, parameter)
    x = contour_detect(imgNoBg)
    xx = blur_image(imgNoBg, x)
    xx = reduce_size(xx, parameter)
    return xx


def detect(img2):
    """
        Function to detect person face in image.
        :param img2: input image
        :return: 1 , if only one face (clear front view image) is visible
                 0 , otherwise
    """

    flag = 0
    global face_position
    image = img2.copy()

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)
    face = []
    for c in faces:
        if (c[2] * c[3]) > 14000:
            face.append(c)

    if len(face) >= 1:
        for (x, y, w, h) in faces:
            # cv2.circle(image, (x + (w // 2), y + (h // 2)), 5, (255, 0, 0), 30)
            roi_gray = gray_image[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 0:
                flag = 1
                face_position = [x, y, w, h]
    return flag


def solve(image, parameter):
    """
        Main function which is called by main.py file
        :param image: input image
        :param parameter: list containing size, padding, background-color information
        :return: final image with changed background
    """
    flag = detect(image)
    if flag == 1:
        final_flag = 0
        imge = remove(image, parameter)
        final_flag = detect_final(imge)
        if final_flag == 0:
            return []
        return imge
    else:
        return []


if __name__ == '__main__':
    img = cv2.imread('i7.jpg')
    size = (480, 480)
    solve(img, size)
    cv2.destroyAllWindows()
