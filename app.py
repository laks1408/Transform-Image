from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)

original_image = None
processed_image = None

#Grayscale Function
def to_grayscale(img):
    img_array = np.array(img)
    grayscale_array = (
        0.299 * img_array[:, :, 0] +
        0.587 * img_array[:, :, 1] +
        0.114 * img_array[:, :, 2]
    )
    return Image.fromarray(grayscale_array.astype('uint8'))

#Blur Function
def apply_blur(img, kernel_size=15): #kernel size = intensitas blur
    img_array = np.array(img)
    if len(img_array.shape) == 2:  #handle grayscale images
        img_array = np.expand_dims(img_array, axis=-1)

    blurred = np.zeros_like(img_array)
    pad = kernel_size // 2

    padded_img = np.pad(img_array, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

    # Apply blur
    for i in range(img_array.shape[0]): #R
        for j in range(img_array.shape[1]): #G
            for c in range(img_array.shape[2]): #B
                region = padded_img[i:i + kernel_size, j:j + kernel_size, c]
                blurred[i, j, c] = np.mean(region)

    return Image.fromarray(blurred.astype('uint8'))

@app.route("/", methods=["GET", "POST"]) #route ke index.html
def index():
    global original_image, processed_image

    if request.method == "POST":
        #upload file
        file = request.files['image']
        option = request.form['option']
        
        if file:
            original_image = Image.open(file)

            if option == "Grayscale":
                processed_image = to_grayscale(original_image)
            elif option == "Blur":
                processed_image = apply_blur(original_image)

            return redirect(url_for("compare"))

    return render_template("index.html")

@app.route("/compare") #route ke compare.html
def compare():
    global original_image, processed_image

    #image rendering html
    original_image_io = BytesIO()
    processed_image_io = BytesIO()

    if original_image:
        original_image.save(original_image_io, format="PNG")
    if processed_image:
        processed_image.save(processed_image_io, format="PNG")

    original_image_io.seek(0)
    processed_image_io.seek(0)

    original_image_data = original_image_io.read()
    processed_image_data = processed_image_io.read()

    original_b64 = base64.b64encode(original_image_data).decode('utf-8')
    processed_b64 = base64.b64encode(processed_image_data).decode('utf-8')

    return render_template("compare.html", original_image=original_b64, processed_image=processed_b64)

if __name__ == "__main__":
    app.run(debug=True)
