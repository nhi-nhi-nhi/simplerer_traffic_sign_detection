import asyncio
import base64
import json
import multiprocessing
import time
from io import BytesIO
from multiprocessing import Process, Queue
import functools
import cv2
import numpy as np
import websockets
from PIL import Image
from logic import CarLogic
from lane_line_detection import *
from traffic_sign_detection import detect_traffic_signs

# Initialize traffic sign classifier
traffic_sign_model = cv2.dnn.readNetFromONNX(r"p2_traffic_sign_detection\traffic_sign_classifier_lenet_v3.onnx")

# Global queue to save the current image
# Used to run the sign classification model in a separate process
g_image_queue = Queue(maxsize=5)

def process_traffic_sign_loop(g_image_queue, signs):
    """Run the sign classification model continuously in a separate process."""
    count_sign = 0
    last_sign = ''
    
    while True:
        if g_image_queue.empty():
            time.sleep(0.1)
            continue
        
        image = g_image_queue.get()
        draw = image.copy()

        # Detect traffic signs
        detected_signs = detect_traffic_signs(image, traffic_sign_model, draw=draw)
        
        if last_sign == '' or last_sign == detected_signs:
            count_sign += 1
        else:
            count_sign = 0
        
        # Update the shared signs list
        signs[:] = detected_signs if count_sign >= 15 else []
        
        # Display the detected signs
        cv2.imshow("Traffic signs", draw)
        cv2.waitKey(1)

async def process_image(websocket, signs):
    """Process incoming images and control the car based on detected lanes and signs."""
    car_control = CarLogic()
    
    async for message in websocket:
        # Decode the image from the received message
        data = json.loads(message)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (640, 480))
        draw = image.copy()

        # Calculate throttle and steering angle
        throttle, steering_angle = car_control.decision_control(image, signs, draw=draw)

        # Add image to the queue for traffic sign detection
        if not g_image_queue.full():
            g_image_queue.put(image)

        # Display the resulting image with lane and sign information
        cv2.imshow("Result", draw)
        cv2.waitKey(1)

        # Send the control commands back to the simulation
        response = json.dumps({"throttle": throttle, "steering": steering_angle})
        await websocket.send(response)

async def main():
    """Main entry point for the WebSocket server."""
    process_image_partial = functools.partial(process_image, signs=signs)
    
    async with websockets.serve(process_image_partial, "0.0.0.0", 4567, ping_interval=None):
        await asyncio.Future()  # Run forever

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    signs = manager.list()
    
    # Start the traffic sign detection process
    p = Process(target=process_traffic_sign_loop, args=(g_image_queue, signs))
    p.start()
    
    # Run the WebSocket server
    asyncio.run(main())