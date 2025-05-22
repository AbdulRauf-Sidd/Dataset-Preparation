import numpy as np;
import cv2

def gamma_correction(image, gamma=1.0):
    # Apply gamma correction
    gamma_corrected = np.uint8(np.clip(((image / 255.0) ** gamma) * 255.0, 0, 255))
    return gamma_corrected

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from PIL import Image
import threading

def augment_images(source_folder_rgb, output_folder_rgb, size, index, rotation, width, height, zoom, resize=True):
    # Data generator for RGB images (with interpolation and rescaling)
    datagen_rgb = ImageDataGenerator(
        rotation_range=rotation,
        width_shift_range=width,
        height_shift_range=height,
        zoom_range=zoom,
        fill_mode='nearest'
    )

    # Ensure output folders exist
    os.makedirs(output_folder_rgb, exist_ok=True)

    # Get all file names from the RGB folder assuming both folders contain the same names
    files = [f for f in os.listdir(source_folder_rgb) if f.endswith('.png') or f.endswith('.jpg')]
    files = files[index[0]: index[1]]
    print(files)
    
    for file in files:
        # Construct paths
        file_path_rgb = os.path.join(source_folder_rgb, file)
        
        # Load and resize images
        img_rgb = Image.open(file_path_rgb).convert('RGB')
        if resize:
            img_rgb = img_rgb.resize((size[0], size[1]))

        # Convert images to arrays
        img_rgb = np.array(img_rgb)
        
        # Reshape for data generator
        img_rgb = img_rgb.reshape((1,) + img_rgb.shape)  # No extra channel dimension needed for RGB
        
        # Generate augmented images
        rgb_gen = datagen_rgb.flow(img_rgb, batch_size=1, seed=42)
        
        for i in range(10):  # Generate 10 augmented images per input image
            batch_rgb = rgb_gen.__next__()
            
            # Save augmented images with consistent naming
            augmented_file_name = f'aug_{i}_{file}'
            Image.fromarray((batch_rgb[0, :, :, 0] * 255).astype('uint8')).save(os.path.join(output_folder_rgb, augmented_file_name))
            

def parallel_process_files(source_rgb, output_rgb, size, rotation, width, height, zoom, resize):
    # Get all files in the specified folder
    files = [f for f in os.listdir(source_rgb) if os.path.isfile(os.path.join(source_rgb, f))]
    total_files = len(files)

    print("Total files:", total_files)
    
    # Determine the chunk size for each thread
    chunk_size = total_files // 8
    remainder = total_files % 8

    threads = []
    start_index = 0

    # Create 8 threads
    for i in range(8):
        # Calculate end index for each chunk
        end_index = start_index + chunk_size + (1 if i < remainder else 0)

        
        # Create a thread with the target function and 5 parameters
        thread = threading.Thread(target=augment_images, args=(source_rgb, output_rgb, size, (start_index, end_index), rotation, width, height, zoom, resize))
        threads.append(thread)
        
        # Start the thread
        thread.start()
        
        # Update start index for the next chunk
        start_index = end_index

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

# Define paths
# source_folder_rgb = '/home/abdulrauf/Projects/MakhiMeter-Training/data/brain/rgb/'
# source_folder_mask = '/home/abdulrauf/Projects/MakhiMeter-Training/data/brain/masked/'
# output_folder_rgb = '/home/abdulrauf/Projects/MakhiMeter-Training/data/brain/rgb aug/'
# output_folder_mask = '/home/abdulrauf/Projects/MakhiMeter-Training/data/brain/masked aug/'