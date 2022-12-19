from flask import Flask, request, send_file

app = Flask(__name__)

@app.route('/check')
def chec():
    return "Hello Hello"

@app.route('/passport')
def edit_passport():
    url = request.args.get('url')
    Img_W = request.args.get('Img_W', type=int)
    Img_H = request.args.get('Img_H', type=int)
    Padding_top = request.args.get('Padding_top', type=int)
    Padding_bottom = request.args.get('Padding_bottom', type=int)
    Padding_left = request.args.get('Padding_left', type=int)
    Padding_right = request.args.get('Padding_right', type=int)
    color_red = request.args.get('color_red', type=int)
    color_green = request.args.get('color_green', type=int)
    color_blue = request.args.get('color_blue', type=int)

    import detect_person as dt
    import cv2 as cv
    import requests
    import shutil


    # STEPS TO DOWNLOAD IMAGE
    res = requests.get(url, stream=True)
    file_name = 'check.jpg'
    if res.status_code == 200:
        with open(file_name, 'wb') as f:
            shutil.copyfileobj(res.raw, f)
        # print('Image sucessfully Downloaded: ', file_name)
        f.close()
    else:
        return ('Image Couldn\'t be retrieved')

    # STEPS TO EDIT IMAGE
    img = cv.imread(file_name)
    if img.shape[0] < 250 or img.shape[1] < 250:
        return "Image size too small resizing it will lead to distoration of image"

    parameter = [Img_W, Img_H, Padding_top, Padding_bottom, Padding_left, Padding_right,color_red,color_green,color_blue]
    edit_image = dt.solve(img, parameter)
    if len(edit_image) == 0:
        return "Invalid Image"
    else:
        cv.imwrite("edited_image.jpg", edit_image)
        import base64
        '''with open("edited_image.jpg", "rb") as img_file:
            b64_string = base64.b64encode(img_file.read())
        return b64_string'''
        # file = open('edited_image.jpg', mode="rb")
        return send_file('edited_image.jpg')


if __name__ == '__main__':
    app.run(debug=True)
