# from flask import Flask, render_template, request
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np
# import io
# import tensorflow as tf
# app = Flask(__name__)

# # Load the deep learning model
# model = tf.keras.models.load_model('models/colorization_generator_model_gan_new_3000.h5')

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # Check if POST request contains file part
#         if 'file' not in request.files:
#             return render_template('base.html', error='No file part')
        
#         file = request.files['file']
        
#         # If user does not select file, browser also
#         # submit an empty part without filename
#         if file.filename == '':
#             return render_template('base.html', error='No selected file')
        
#         if file:
#             # Read the uploaded image
#             img = Image.open(io.BytesIO(file.read()))
            
#             # Resize image to match model input shape
#             # img = img.resize((256, 256))
            
#             # Convert image to numpy array
#             # img_array = np.array(img) / 255.0
            
#             # Add batch dimension
#             # img_array = np.expand_dims(img_array, axis=0)
            
#             # Generate colored image using the model
#             # colored_img = model.predict(img_array)[0]
#             colored_img = model.predict(img)
#             colored_img = (colored_img * 255).astype(np.uint8)
            
#             # Convert numpy array to PIL Image
#             colored_img_pil = Image.fromarray(colored_img)
            
#             # Save the colored image temporarily
#             temp_img_path = 'static/temp_colored_img.jpg'
#             colored_img_pil.save(temp_img_path)
            
#             # Pass the temporary path to the template for display
#             return render_template('base.html', img_path=temp_img_path)

#     return render_template('base.html')

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, render_template, request, send_file
# from keras.models import load_model
# import numpy as np
# from PIL import Image

# app = Flask(__name__,static_folder='static')
# model_path = 'models/colorization_generator_model_gan_new_3000.h5'

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # Check if the post request has the file part
#         if 'file' not in request.files:
#             return render_template('index.html', error='No file part')
#         file = request.files['file']
#         # If the user does not select a file, the browser submits an empty file without a filename
#         if file.filename == '':
#             return render_template('index.html', error='No selected file')
#         # If a file is uploaded and it's an image
#         if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             # Load the GAN model
#             gan_model = load_model(model_path)
#             # Read the uploaded image
#             img = Image.open(file)
#             # Perform colorization prediction (example: just resizing image)
#             colorized_img = img.resize((256, 256))
#             # Save the colorized image to a temporary file
#             colorized_img_path = 'temp_colorized_image.jpg'
#             colorized_img.save(colorized_img_path)
#             # Return the colorized image
#             return render_template('base.html', colorized_img=colorized_img_path)
#         else:
#             return render_template('base.html', error='Invalid file format')
#     return render_template('base.html')

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, redirect, render_template,url_for, request
from keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import shutil
import glob
import random
import os
import math
import itertools
import sys
from skimage import color
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image
import argparse# Importing Libraries

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision.models import vgg19


app = Flask(__name__, static_folder='static')
model_path1='models\colorization_generator_model_gan_new_3000.h5'
model_path2 = r'models\unetmodel.h5'
model_path3=r'models\gan_new_3000.h5'

def preprocess_grayscale_image(image_path, size):
    size=160
    # Read the image
    grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize the image
    grayscale_image = cv2.resize(grayscale_image, (size, size))
    # Convert grayscale to RGB
    grayscale_image_rgb = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)
    # Normalize the image
    grayscale_image_rgb = grayscale_image_rgb.astype('float32') / 255.0
    # Reshape the image to match model input shape
    grayscale_image_rgb = grayscale_image_rgb.reshape(1, size, size, 3)
    return grayscale_image_rgb

SIZE=160
      

def predict_new_image(model, grayscale_image):
    # Make prediction
    predicted_image = model.predict(grayscale_image)
    # Clip the values to be between 0 and 1
    predicted_image = np.clip(predicted_image, 0.0, 1.0)
    # Reshape the predicted image
    # Ensure that the shape is appropriate for your model output
    predicted_image = predicted_image.reshape(SIZE, SIZE, 3)
    # Convert the predicted image array to PIL Image
    predicted_image_pil = Image.fromarray((predicted_image * 255).astype(np.uint8))
    return predicted_image_pil

# ********model 3************
def predict_new_image1(model, grayscale_image):
    SIZE=256
    # Make prediction
    predicted_image = model.predict(grayscale_image)
    # Clip the values to be between 0 and 1
    predicted_image = np.clip(predicted_image, 0.0, 1.0)
    # Reshape the predicted image
    # Ensure that the shape is appropriate for your model output
    predicted_image = predicted_image.reshape(SIZE, SIZE, 3)
    # Convert the predicted image array to PIL Image
    predicted_image_pil = Image.fromarray((predicted_image * 255).astype(np.uint8))
    return predicted_image_pil


def preprocess_image_model1(image_path, size):
    # Read the image
    image = cv2.imread(image_path)
    # Resize the image
    resized_image = cv2.resize(image, (256, 256))
    # Convert the image to RGB (assuming it's in BGR format)
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    # Normalize the image
    normalized_image = rgb_image.astype('float32') / 255.0
    # Reshape the image to match model input shape
    reshaped_image = normalized_image.reshape(1, 256, 256, 3)
    return reshaped_image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('base.html', error='No file part')
        file = request.files['file']
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):            
            img = Image.open(file)
        if file.filename == '':
            return render_template('base.html', error='No selected file')
        grayscale_img_path = 'static/temp_grayscale_image.jpg'
        img.save(grayscale_img_path)

        # if "model1" in request.form:
        #     gan_model = load_model(model_path1)
        #     colorized_img = colorize_image(img, gan_model)
        #     colorized_img_path = 'static/gan_colorized_image.jpg'
        #     colorized_img.save(colorized_img_path)
        #     return redirect(url_for('result', grayscale_img=grayscale_img_path, colorized_img=colorized_img_path))
        
        if "model1" in request.form:
            gan_model = load_model(model_path1)
            new_grayscale_img = preprocess_image_model1(grayscale_img_path, size=256)
            predicted_image = predict_new_image1(gan_model,new_grayscale_img)
            colorized_img_path = 'static/gan_colorized_image.jpg'
            predicted_image.save(colorized_img_path)
            return redirect(url_for('result', grayscale_img=grayscale_img_path, colorized_img=colorized_img_path))
        
        elif "model2" in request.form:
            unet_model = load_model(model_path2)
            new_grayscale_img=preprocess_grayscale_image(r"static/temp_grayscale_image.jpg",SIZE)
            predicted_image=predict_new_image(unet_model,new_grayscale_img)
            colorized_img_path = r'static/unet_colorized_image.jpg'
            predicted_image.save(colorized_img_path)
            return redirect(url_for('result', grayscale_img=grayscale_img_path, colorized_img=colorized_img_path))
        
        elif "model3" in request.form:
            print("helloo")
            class BaseColor(nn.Module):
                def __init__(self):
                    super(BaseColor, self).__init__()

                    self.l_cent = 50.
                    self.l_norm = 100.
                    self.ab_norm = 110.

                def normalize_l(self, in_l):
                    return (in_l-self.l_cent)/self.l_norm

                def unnormalize_l(self, in_l):
                    return in_l*self.l_norm + self.l_cent

                def normalize_ab(self, in_ab):
                    return in_ab/self.ab_norm

                def unnormalize_ab(self, in_ab):
                    return in_ab*self.ab_norm


            class ECCVGenerator(BaseColor):
                def __init__(self, norm_layer=nn.BatchNorm2d):
                    super(ECCVGenerator, self).__init__()

                    model1=[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),]
                    model1+=[nn.ReLU(True),]
                    model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
                    model1+=[nn.ReLU(True),]
                    model1+=[norm_layer(64),]

                    model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
                    model2+=[nn.ReLU(True),]
                    model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
                    model2+=[nn.ReLU(True),]
                    model2+=[norm_layer(128),]

                    model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
                    model3+=[nn.ReLU(True),]
                    model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
                    model3+=[nn.ReLU(True),]
                    model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
                    model3+=[nn.ReLU(True),]
                    model3+=[norm_layer(256),]

                    model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
                    model4+=[nn.ReLU(True),]
                    model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
                    model4+=[nn.ReLU(True),]
                    model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
                    model4+=[nn.ReLU(True),]
                    model4+=[norm_layer(512),]

                    model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
                    model5+=[nn.ReLU(True),]
                    model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
                    model5+=[nn.ReLU(True),]
                    model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
                    model5+=[nn.ReLU(True),]
                    model5+=[norm_layer(512),]

                    model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
                    model6+=[nn.ReLU(True),]
                    model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
                    model6+=[nn.ReLU(True),]
                    model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
                    model6+=[nn.ReLU(True),]
                    model6+=[norm_layer(512),]

                    model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
                    model7+=[nn.ReLU(True),]
                    model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
                    model7+=[nn.ReLU(True),]
                    model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
                    model7+=[nn.ReLU(True),]
                    model7+=[norm_layer(512),]

                    model8=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
                    model8+=[nn.ReLU(True),]
                    model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
                    model8+=[nn.ReLU(True),]
                    model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
                    model8+=[nn.ReLU(True),]

                    model8+=[nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]

                    self.model1 = nn.Sequential(*model1)
                    self.model2 = nn.Sequential(*model2)
                    self.model3 = nn.Sequential(*model3)
                    self.model4 = nn.Sequential(*model4)
                    self.model5 = nn.Sequential(*model5)
                    self.model6 = nn.Sequential(*model6)
                    self.model7 = nn.Sequential(*model7)
                    self.model8 = nn.Sequential(*model8)

                    self.softmax = nn.Softmax(dim=1)
                    self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
                    self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

                def forward(self, input_l):
                    conv1_2 = self.model1(self.normalize_l(input_l))
                    conv2_2 = self.model2(conv1_2)
                    conv3_3 = self.model3(conv2_2)
                    conv4_3 = self.model4(conv3_3)
                    conv5_3 = self.model5(conv4_3)
                    conv6_3 = self.model6(conv5_3)
                    conv7_3 = self.model7(conv6_3)
                    conv8_3 = self.model8(conv7_3)
                    out_reg = self.model_out(self.softmax(conv8_3))

                    return self.unnormalize_ab(self.upsample4(out_reg))

            """def eccv16(pretrained=True):
                model = ECCVGenerator()
                if(pretrained):
                    import torch.utils.model_zoo as model_zoo
                    model.load_state_dict(torch.load('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',map_location=torch.device('cpu'),check_hash=True))
                return model"""
            def eccv16(pretrained=True):
                model = ECCVGenerator()
                if pretrained:
                    file_path = 'colorization_release_v2-9b330a0b.pth'  # Provide the local file path
                    model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))  
                return model

            class color_ecv(nn.Module):
                def __init__(self, in_channels=3):
                    super(color_ecv, self).__init__()
                    
                    self.model = eccv16(pretrained=True)
                
                def forward(self, x):
                    ecv_output = self.model(x)
                    return ecv_output
                
            def load_img(img_path):
                out_np = np.asarray(Image.open(img_path))
                if(out_np.ndim==2):
                    out_np = np.tile(out_np[:,:,None],3)
                return out_np

            def resize_img(img, HW=(256,256), resample=3):
                return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

            def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
                # return original size L and resized L as torch Tensors
                img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
                
                img_lab_orig = color.rgb2lab(img_rgb_orig)
                img_lab_rs = color.rgb2lab(img_rgb_rs)

                img_l_orig = img_lab_orig[:,:,0]
                img_l_rs = img_lab_rs[:,:,0]

                tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]
                tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]

                return (tens_orig_l, tens_rs_l)

            def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
                # tens_orig_l 	1 x 1 x H_orig x W_orig
                # out_ab 		1 x 2 x H x W

                HW_orig = tens_orig_l.shape[2:]
                HW = out_ab.shape[2:]

                # call resize function if needed
                if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
                    out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
                else:
                    out_ab_orig = out_ab

                out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
                return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))

            def postprocess_tens_new(tens_orig_l, out_ab, mode='bilinear'):
                # tens_orig_l 	Batchsize x 1 x H_orig x W_orig
                # out_ab 		Batchsize x 2 x H x W
                Batchsize = tens_orig_l.shape[0]

                output_ = []
                for i in range(Batchsize):
                    tens_orig_l_i = tens_orig_l[i][np.newaxis, :, :, :]
                    out_ab_i  = out_ab[i][np.newaxis, :, :, :]
                    HW_orig_i = tens_orig_l_i.shape[2:]
                    HW_i = out_ab_i.shape[2:]

                    # call resize function if needed
                    if(HW_orig_i[0]!=HW_i[0] or HW_orig_i[1]!=HW_i[1]):
                        out_ab_orig_i = F.interpolate(out_ab_i, size=HW_orig_i, mode='bilinear')
                    else:
                        out_ab_orig_i = out_ab_i

                    out_lab_orig_i = torch.cat((tens_orig_l_i, out_ab_orig_i), dim=1)
                    #output_.append(color.lab2rgb(out_lab_orig_i.data.cpu().numpy()[0,...].transpose((1,2,0))))
                    output_.append(color.lab2rgb(out_lab_orig_i.data.cpu().numpy()[0,...].transpose((1,2,0))).transpose((2,0,1)))
                return np.array(output_)


            model = color_ecv()
            model.load_state_dict(torch.load("pretrained_models/generator.pth", map_location=torch.device('cpu')))
            model.eval()

            class TestDataset(Dataset):
                def __init__(self, root, single_image):
                    if single_image:
                        self.files = [root]
                    else:
                        self.files = sorted(glob.glob(root + "/*.*"))
                    
                def __getitem__(self, index):
                
                    black_path = self.files[index % len(self.files)]
                    img_black = np.asarray(Image.open(black_path))
                    if(img_black.ndim==2):
                        img_black = np.tile(img_black[:,:,None],3)
                    (tens_l_orig, tens_l_rs) = preprocess_img(img_black, HW=(400, 400))

                    return {"black": tens_l_rs.squeeze(0), 'orig': tens_l_orig.squeeze(0), 'path' : black_path}
                
                def __len__(self):
                    return len(self.files)
                
            def predict_outputs(model, dataset):
                #image = single_image
                batch_size = 1
                dataloader = DataLoader(
                    dataset,
                    batch_size = batch_size,
                    shuffle = False,
                    num_workers = 0,
                )

                # cuda = torch.cuda.is_available()
                # if cuda:
                #     model = model.to('cuda')

                # Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
                outputs = {}
                for i, imgs in enumerate(dataloader):
                    # imgs_black = Variable(imgs["black"].type(Tensor))
                    imgs_black = Variable(imgs["black"])
                    # imgs_black_orig = Variable(imgs["orig"].type(Tensor))
                    imgs_black_orig = Variable(imgs["orig"])
                    gen_ab = model(imgs_black)
                    gen_ab.detach_()
                    gen_color = postprocess_tens_new(imgs_black_orig, gen_ab)[0].transpose(1,2,0)
                    outputs[imgs["path"][0]] = gen_color
                return outputs

            def print_images(outputs):
                for i in outputs.keys():
                    print("----------- The Black and White Image -----------")
                    plt.imshow(plt.imread(i))
                    plt.show()
                    print("----------- The Colourfull Image Generated -----------")
                    plt.imshow(outputs[i])
                    plt.show()

            def save_outputs(outputs, folder_path, single_image):
                os.makedirs(folder_path,  exist_ok=True)
                for i in outputs.keys():
                    if single_image:
                        name = i.split('/')[-1]
                    else:
                        name = i.split('\\')[-1]
                    image = Image.fromarray((outputs[i] * 255).astype(np.uint8)) 
                    image.save(folder_path + '/' + name)

            single_image = True
            input_path=r"D:\imagecolorproject\static\temp_grayscale_image.jpg"
            source_path=r"D:\imagecolorproject\static\input.jpg"
            shutil.copy(input_path,source_path)
            path ="static/input.jpg"
            dataset = TestDataset(path, single_image)
            outputs = predict_outputs(model, dataset)

            #print_images(outputs)

            outputs = predict_outputs(model, dataset)
            outputs['static/model3.jpg'] = outputs[dataset.files[0]]  # Assuming the first file in the dataset is 'model3.jpg'
            save_outputs(outputs, folder_path='static/', single_image=True)
            colorized_img_path="static\model3.jpg"
            return redirect(url_for('result', grayscale_img=grayscale_img_path, colorized_img=colorized_img_path))
            """single_image = False
            path = 'sample'
            dataset = TestDataset(path, single_image)
            outputs = predict_outputs(model, dataset)

            print_images(outputs)

            save_outputs(outputs, folder_path = 'Outputs/multiple_Outputs', single_image = single_image)"""

             
        
        elif "compare" in request.form:
            compare()
        elif file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):            
            # Load the model
            img = Image.open(file)
            gan_model = load_model(model_path1)
            unet_model = load_model(model_path2)
            base_ganmodel=load_model(model_path3)
            grayscale_img_path = 'static/temp_grayscale_image.jpg'
            img.save(grayscale_img_path)
            #model1
            colorized_img = colorize_image(img, gan_model)
            # Save the colorized image to a temporary file
            colorized_img_path = 'static/gan_colorized_image.jpg'
            colorized_img.save(colorized_img_path)
            # Return the colorized image
            # return render_template('base.html', colorized_img=colorized_img_path)
            # Save the grayscale input image to a temporary file
            grayscale_img_path = 'static/temp_grayscale_image.jpg'
            img.save(grayscale_img_path)

            #model2
            new_grayscale_img=preprocess_grayscale_image("static/temp_grayscale_image.jpg",SIZE)
            predicted_image=predict_new_image(unet_model,new_grayscale_img)
            new_grayscale_img=colorize_image("static/temp_grayscale_image.jpg",SIZE)
            colorized_img_path = 'static/unet_colorized_image.jpg'
            predicted_image.save(colorized_img_path)

            #model3
            colorized_img= colorize_image(img, base_ganmodel)
            # Save the colorized image to a temporary file
            colorized_img_path = 'static/basegan_colorized_image.jpg'
            colorized_img.save(colorized_img_path)
            # Return the colorized image
            # return render_template('base.html', colorized_img=colorized_img_path)
            # Save the grayscale input image to a temporary file
        

            # Read the uploaded image
        
            
            # Redirect to the result page with the paths of both images as query parameters"""
            return redirect(url_for('result', grayscale_img=grayscale_img_path, colorized_img=colorized_img_path))
        else:
            return render_template('base.html', error='Invalid file format')
    return render_template('base.html')



@app.route('/result')
def result():
    grayscale_img_path = request.args.get('grayscale_img')
    colorized_img_path = request.args.get('colorized_img')
    return render_template('result.html', grayscale_img=grayscale_img_path, colorized_img=colorized_img_path)

@app.route("/compare",methods=["POST"])
def compare():
    if "compare" in request.form:
        """gan_output = request.args.get('colorized_img')
        unet_output = request.args.get('predicted_image')"""
        upload_path=r"static\temp_grayscale_image.jpg"
        gan_path=r"static\gan_colorized_image.jpg"
        unet_path=r"static\unet_colorized_image.jpg"
        basegan_path=r"static\model3.jpg"
        
        return render_template('comparison.html', upload_path=upload_path,gan_path=gan_path,unet_path=unet_path,basegan_path=basegan_path)
    else:
        return render_template('base.html')
    
# def colorize_image(img, model):
#     # Resize image to desired input shape of the model
#     img = img.resize((256, 256))
#     # Convert image to numpy array and add color channel dimension
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=-1)  # Add color channel dimension
#     # Repeat color channel to match the expected input shape of the model
#     img_array = np.repeat(img_array, 3, axis=-1)  # Repeat grayscale channel to RGB
#     # Normalize pixel values (assuming pixel range is [0, 255])
#     img_array = img_array / 255.0
#     # Expand dimensions to match model input shape (if needed)
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     # Perform colorization prediction using the model
#     colorized_img_array = model.predict(img_array)
#     # Post-process the colorized image array if needed
#     # Convert the colorized image array back to PIL Image
#     colorized_img = Image.fromarray((colorized_img_array[0] * 255).astype(np.uint8))
#     return colorized_img


def colorize_image(img, model):
    # Resize image to desired input shape of the model
    img = img.resize((256, 256))
    # Convert image to numpy array
    img_array = np.array(img)
    # Convert grayscale image to RGB by stacking the grayscale image three times
    img_array = np.stack((img_array,) * 3, axis=-1)
    # Normalize pixel values (assuming pixel range is [0, 255])
    img_array = img_array / 255.0
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    # Perform colorization prediction using the model
    colorized_img_array = model.predict(img_array)
    # Post-process the colorized image array if needed
    # Convert the colorized image array back to PIL Image
    colorized_img = Image.fromarray((colorized_img_array[0] * 255).astype(np.uint8))
    return colorized_img



# def preprocess_input_basegan(image_path, img_size):
#     img_size=120
#     # Read and resize the input grayscale image
#     input_gray_image = Image.open(image_path).resize((img_size, img_size))
#     # Convert to grayscale
#     input_gray_image = input_gray_image.convert('L')
#     # Normalize the grayscale image array to 0-1
#     input_gray_img_array = np.asarray(input_gray_image).reshape((1, img_size, img_size, 1)) / 255
#     return input_gray_img_array






#model 3



if __name__ == '__main__':
    app.run(debug=True)
