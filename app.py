from flask import Flask,render_template,request,redirect
import os

#Using a VGG16 model to predict
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import ResNet50
model = ResNet50()


app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def home():
    if request.method == 'POST':
        if request.files:            
            image = request.files['image']

            if image.filename == '':
                print('Image must have a filename')
                return redirect(request.url)

            image.save(os.path.join('static',image.filename))
            print('Image saved')
            
            image_path = "static/" + image.filename
            
            pred_image = load_img(image_path,target_size=(224,224))
            pred_image = img_to_array(pred_image)
            pred_image = pred_image.reshape((1,pred_image.shape[0],pred_image[1],pred_image[2]))
            pred_image = preprocess_input(pred_image)
            out = model.predict(pred_image)
            label = decode_predictions(out)
            label = label[0][0]
            result = '%s (%.2f%%)' % (label[1],label[2]*100)

            return render_template('upload.html',prediction=result)

    return render_template('upload.html')


if __name__ == "__main__":
    app.run(debug=True)
