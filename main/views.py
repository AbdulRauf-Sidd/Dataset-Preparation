from django.shortcuts import render
import cv2
import os
import numpy as np
from .helpers import gamma_correction
import base64
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# def image(request):
#     if request.method == 'POST':
#         image = cv2.imread('main/static/images/preview.jpg', cv2.IMREAD_GRAYSCALE)
#         max_intensity = int(request.POST.get('value', 255))
#         print(max_intensity)
#         equalized_image = cv2.equalizeHist(image)
#         scaled_image = (equalized_image / image[0]*image[1]) * max_intensity 
#         scaled_image = np.clip(scaled_image, 0, max_intensity).astype(np.uint8)
        
#         hist = cv2.calcHist([scaled_image], [0], None, [256], [0, 256])

#         # Plot the histogram
#         fig = Figure()
#         ax = fig.add_subplot(111)
#         ax.plot(hist)
#         ax.set_xlim([0, 255])
#         ax.set_title('Histogram')
#         ax.set_xlabel('Intensity')
#         ax.set_ylabel('Frequency')

#         # Render the histogram plot to a PNG image
#         canvas = FigureCanvas(fig)
#         response = HttpResponse(content_type='image/png')
#         canvas.print_png(response)
        
#     return render(request, 'img.html');

image='image0115.png'
path = 'main/static/images/' + image;
output = 'main/static/images/output/' +image

def gamma(request):
    return render(request, 'gamma.html');


def update(request):
    gamma_val = float(request.GET.get('gamma', 1))
    threshold_val = int(request.GET.get('threshold', 45))
    dilation_val = int(request.GET.get('dilation', 1))
    erosion_val = int(request.GET.get('erosion', 1))
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE);
    gamma_value = 1.0/float(gamma_val)
        
    image = gamma_correction(image, gamma_value);
    _, binary_image = cv2.threshold(image, threshold_val, 255, cv2.THRESH_BINARY)
    inverted_image = cv2.bitwise_not(binary_image)

    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(inverted_image, kernel, iterations=dilation_val)

    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=erosion_val)

    # Save the processed image
    cv2.imwrite(output, eroded_image)
        
    # cv2.imwrite(output, inverted_image);
        
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        
    frequency = [];
    for i in range(len(hist)):
        frequency.append(hist[i]);
    
    image = cv2.imread(output)
    # print(image)

    _, encoded_image = cv2.imencode('.jpg', image)
    encoded_image_str = base64.b64encode(encoded_image).decode('utf-8')
    
    
    serializable_list = [arr.tolist()[0] for arr in frequency]
    return JsonResponse({'image': str(encoded_image_str), 'histo':serializable_list})

def histogram_equalization(request):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #equalized_image = cv2.equalizeHist(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    cl1 = clahe.apply(image)
    
    cv2.imwrite(output, cl1);
    
    hist = cv2.calcHist([cl1], [0], None, [256], [0, 256]);
    
    frequency = [];
    for i in range(len(hist)):
        frequency.append(hist[i]);
    
    image = cv2.imread(output)
    _, encoded_image = cv2.imencode('.jpg', image)
    encoded_image_str = base64.b64encode(encoded_image).decode('utf-8')
    
    
    serializable_list = [arr.tolist()[0] for arr in frequency]
    return JsonResponse({'image': str(encoded_image_str), 'histo':serializable_list})
    
    
    