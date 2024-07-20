import threading
import capture
import detect
import logging
import os

# Suppress TensorFlow and Mediapipe logging
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (0=all logs, 1=info, 2=warning, 3=error)
#logging.getLogger('tensorflow').setLevel(logging.FATAL)

save_dir = 'captured_images'

if __name__ == "__main__":
    capture_images_thread = threading.Thread(target=capture.capture_images, args=(save_dir,))
    capture_images_thread.start()
    
    detect.monitor_images(save_dir)
