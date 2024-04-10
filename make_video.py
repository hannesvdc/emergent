import os
import cv2

def save_pdfdd():
    image_folder = 'fig'
    video_name = 'video_pdfdd.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png") and img.startswith('pdfdd_')]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def save_sle():
    image_folder = 'fig'
    video_name = 'video_sle.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png") and img.startswith('sle_')]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def save_pdf3d_proj():
    image_folder = 'fig'
    video_name = 'video_pdf3d_proj.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png") and img.startswith('pdf3d_')]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    save_pdfdd()