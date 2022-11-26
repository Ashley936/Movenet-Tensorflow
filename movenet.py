import tensorflow as tf
import numpy as np 
import cv2

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def get_keypoints(image):
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, 192, 192)

    model_path = "lite-model_movenet_singlepose_lightning_3.tflite"
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    input_image = tf.cast(input_image, dtype=tf.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    """ ESTIMATING USING MOVENET """
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])
    return keypoints

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1) 

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

def render_video(): 
    """ Get video """
    cap = cv2.VideoCapture("input_video.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        image = frame.copy()
        
        keypoints = get_keypoints(image)

        """ Rendering keypoints """ 
        draw_connections(frame, keypoints, EDGES, 0.4)
        draw_keypoints(frame, keypoints, 0.4)
        
        cv2.imshow('MoveNet Lightning', frame)
        
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def render_image():
    """ Get Image """
    image_path = "input_image.jpeg"
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
      
    keypoints = get_keypoints(image)
    """ Print keypoints on console """
    print(keypoints)
        
choice = input('Select media-type: \n1.Video 2.Image \n')
if(choice=='1'):
    render_video()
elif (choice=='2'):
    render_image()
else: print('WRONG CHOICE !! Please enter 1 or 2')