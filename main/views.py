from django.shortcuts import render
import cv2
import os
from PIL import Image
# from rembg import remove
import numpy as np
from .helpers import gamma_correction
import base64
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import json
# import csrf_exempt
from django.views.decorators.csrf import csrf_exempt

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

image='top left_bg_remove.png'
path = '/home/abdulrauf/Projects/Makhi Meter 2.0/samples/histo/main/static/images/top left_bg_remove.png';
output = 'main/static/images/output/' +image

def home(request):
    return render(request, 'index.html');

@csrf_exempt
def histogram_view(request):
    if request.method == 'GET':
        image = cv2.imread(os.path.join(settings.STATICFILES_DIRS[0], 'images', 'test.png'), cv2.IMREAD_COLOR)
        _, encoded_image = cv2.imencode('.jpg', image)
        encoded_image_str = base64.b64encode(encoded_image).decode('utf-8')
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
        frequency = [];
        for i in range(len(hist)):
            frequency.append(hist[i]);
        # Now 'image' is a valid OpenCV image (numpy array)
        # You can process the image as needed here
        serializable_list = [arr.tolist()[0] for arr in frequency]

        data = {
            'image': str(encoded_image_str),
            'histo': serializable_list,
            'image2': str(encoded_image_str),
        }

        context = {
            'data': json.dumps(data)
        }
        return render(request, 'histogram.html', context);
    if request.method == 'PUT':
        print('hi')
        try:
            gamma_val = float(request.GET.get('gamma', 1))
            image_org = cv2.imread(os.path.join(settings.STATICFILES_DIRS[0], 'images', 'test.png'), cv2.IMREAD_COLOR)
            print('image_org', image_org.shape)
            gamma_value = 1.0/float(gamma_val)
            image = cv2.resize(image_org, (250, 250), interpolation=cv2.INTER_CUBIC)
            # image.resize(256, 256)
            image = gamma_correction(image, gamma_value);

            image = cv2.resize(image, (image_org.shape[1], image_org.shape[0]), interpolation=cv2.INTER_CUBIC)

            _, encoded_image = cv2.imencode('.jpg', image)
            encoded_image_str = base64.b64encode(encoded_image).decode('utf-8')
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        
            frequency = [];
            for i in range(len(hist)):
                frequency.append(hist[i]);
            # Now 'image' is a valid OpenCV image (numpy array)
            # You can process the image as needed here
            serializable_list = [arr.tolist()[0] for arr in frequency]
            return JsonResponse({'image': str(encoded_image_str), 'histo':serializable_list})
        except Exception as e:
            return JsonResponse({'error': 'No image uploaded'}, status=400)
            # print('error', e)
            # pass
    if request.method == 'POST':
        # print('hi')
        try:
            gamma_val = float(request.GET.get('gamma', 1))
            image_org = cv2.imread(os.path.join(settings.STATICFILES_DIRS[0], 'images', 'test.png'), cv2.IMREAD_COLOR)
            gamma_value = 1.0/float(gamma_val)
            # image = cv2.resize(image_org, (250, 250), interpolation=cv2.INTER_CUBIC)
            # image.resize(256, 256)
            image = gamma_correction(image_org, gamma_value);

            # image = cv2.resize(image, (image_org.shape[0], image_org.shape[1]), interpolation=cv2.INTER_CUBIC)
            # image_org = cv2.imread(os.path.join(settings.STATICFILES_DIRS[0], 'images', 'test.png'), cv2.IMREAD_COLOR)
            images_dir = os.path.join(settings.STATICFILES_DIRS[0], 'images', 'saved')
            os.makedirs(images_dir, exist_ok=True)

            # Save the image to the static directory
            # image_path = os.path.join(images_dir, 'test.png')
            file_path = os.path.join(images_dir, "saved_image.png")
            cv2.imwrite(file_path, image)
            
            return JsonResponse({'image': 'successfully saved', 'path': file_path}, status=200)
        except Exception as e:
            print('error', e)
            return JsonResponse({'error': 'No image uploaded'}, status=400)
            # print('error', e)
            # pass
        
    # return render(request, 'index.html');

def histogram(request):
    if request.method == 'GET':
        return render(request, 'histogram_input.html');

def gamma(request):
    return render(request, 'gamma.html');

def hough(request):
    return render(request, 'hough.html');


def update(request):
    gamma_val = float(request.GET.get('gamma', 1))
    threshold_val = int(request.GET.get('threshold', 45))
    dilation_val = int(request.GET.get('dilation', 1))
    erosion_val = int(request.GET.get('erosion', 1))
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE);
    print(image)
    # image = remove(image)
    # image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    gamma_value = 1.0/float(gamma_val)
    # image.resize(256, 256)
    image = gamma_correction(image, gamma_value);
    _, binary_image = cv2.threshold(image, threshold_val, 255, cv2.THRESH_BINARY)
    inverted_image = cv2.bitwise_not(binary_image)

    kernel = np.ones((2, 2), np.uint8)
    dilated_image = cv2.dilate(inverted_image, kernel, iterations=dilation_val)

    kernel = np.ones((2, 2), np.uint8)
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


def hough_circles(request):
    # Retrieve parameters from GET request with default values
    dp = float(request.GET.get('dp', 1.2))  # Inverse ratio of the accumulator resolution to the image resolution
    minDist = float(request.GET.get('minDist', 100))  # Minimum distance between the centers of the detected circles
    param1 = float(request.GET.get('param1', 50))  # Higher threshold for the Canny edge detector
    param2 = float(request.GET.get('param2', 30))  # Threshold for center detection
    minRadius = int(request.GET.get('minRadius', 0))  # Minimum circle radius
    maxRadius = int(request.GET.get('maxRadius', 0))  # Maximum circle radius

    # Load an image
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print('dajskldjaslkdjsakldjaslkdjsa', image)

    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (255, 0, 0), 2)
            # Draw the center of the circle
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Encode the processed image to send as a response
    _, encoded_image = cv2.imencode('.jpg', image)
    encoded_image_str = base64.b64encode(encoded_image).decode('utf-8')

    # Convert circle data to serializable format if circles are detected
    circles_list = circles.tolist() if circles is not None else []

    return JsonResponse({'image': encoded_image_str, 'circles': circles_list})

def histogram_equalization(request):
    image = cv2.imread(output, cv2.IMREAD_GRAYSCALE)
    #equalized_image = cv2.equalizeHist(image)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    # cl1 = clahe.apply(image)
    image = cv2.resize(image, (256, 256))
    cv2.imwrite(output, image);
    
    # hist = cv2.calcHist([cl1], [0], None, [256], [0, 256]);
    hist = []
    frequency = [];
    for i in range(len(hist)):
        frequency.append(hist[i]);
    
    image = cv2.imread(output)
    _, encoded_image = cv2.imencode('.jpg', image)
    encoded_image_str = base64.b64encode(encoded_image).decode('utf-8')
    
    
    serializable_list = [arr.tolist()[0] for arr in frequency]
    return JsonResponse({'image': str(encoded_image_str), 'histo':serializable_list})
    
    
import os
import numpy as np
import cv2
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def upload_image(request):
    print('request', request)
    print('request.FILES', request.FILES)
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Ensure the images directory exists
        images_dir = os.path.join(settings.STATICFILES_DIRS[0], 'images')
        os.makedirs(images_dir, exist_ok=True)

        # Save the image to the static directory
        image_path = os.path.join(images_dir, 'test.png')
        cv2.imwrite(image_path, image)

        # Return a JSON response
        return JsonResponse({'status': 'success'}, status=200)

    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)


from django.shortcuts import render
from django.http import JsonResponse
from .helpers import augment_images, parallel_process_files

def augmentation(request):
    if request.method == 'GET':
        return render(request, 'augmentation.html')
    if request.method == 'POST':
        rotation_range = request.POST.get('rotation_range')
        width_shift_range = request.POST.get('width_shift_range')
        height_shift_range = request.POST.get('height_shift_range')
        zoom_range = request.POST.get('zoom_range')
        width = request.POST.get('width')
        height = request.POST.get('height')
        resize = request.POST.get('resize')

        input_folder = os.path.join(settings.DATA_DIR, 'input')
        output_folder = os.path.join(settings.DATA_DIR, 'output')

        parallel_process_files(input_folder, output_folder, [int(width), int(height)], float(rotation_range), float(width_shift_range), float(height_shift_range), float(zoom_range), resize)
        

        print("Rotation Range:", rotation_range)
        print("Width Shift Range:", width_shift_range)
        print("Height Shift Range:", height_shift_range)
        print("Zoom Range:", zoom_range)
        

        return JsonResponse({'status': 'success', 'message': 'Inputs received successfully!'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'})