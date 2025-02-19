import asyncio
from contextlib import asynccontextmanager
import json
import shlex
import shutil
import subprocess as sp
from datetime import datetime, timedelta
import os
import time
import traceback

import cv2

from threading import Lock, Thread, Event


# from imageai.Detection import VideoObjectDetection
import aiortc

from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer, VideoStreamTrack
from av import VideoFrame

import requests
import sounddevice as sd
import fractions

from typing import Optional, Set, Tuple

from fastapi import FastAPI, Response, WebSocket, HTTPException, Request, Form
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates
import starlette.status as status


import jwt

from datetime import datetime, timedelta
import bcrypt
import secrets
from typing import Optional
import httpx

import numpy as np


from performance import bandwidth


import myLogger
local_logger = myLogger.get_logger(__name__)

from ultralytics import YOLO
from openvino.runtime import Core
# Suppress ultralytics logging
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)


admin = 'admin'
pwd = 'Brih22031981'

CAMERA_URLS = {
    "camera1": "rtsp://{0}:{1}@192.168.11.61/stream2".format(admin, pwd),
    "camera2": "rtsp://{0}:{1}@192.168.11.62/stream2".format(admin, pwd),
    "camera3": "rtsp://{0}:{1}@192.168.11.63/stream2".format(admin, pwd),
    "camera5": 0,
    "camera4": "/dev/video5",
}


CAMERA_CAP = {}

def truncate_log():
    try:
        # Use the truncate command to clear the log file
        sp.run("reset", shell=1)
        sp.run(["truncate", "-s", "0", 'log/*.log'], check=True)
        local_logger.info(f"Log file truncated at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    except sp.CalledProcessError as e:
        local_logger(f"Failed to truncate log file: {e}")

def start_truncate_schedule():
    while True:
        truncate_log()
        time.sleep(8 * 60 * 60)
        # Truncate the log every 8 hours (28,800 seconds)
        truncate_log()

Thread(target=start_truncate_schedule).start()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    local_logger.info("Brih MediaStream Application is running...")
    
    try:
        yield  # This line separates startup from shutdown logic
    finally:
        # Shutdown logic
        # close peer connections
        coros = [pc.close() for pc in pcs]
        await asyncio.gather(*coros)
        pcs.clear()
        
        for _, cap in CAMERA_CAP.items():
            if cap:
                cap.stop_capture()
        local_logger.warning("Brih MediaStream Application is Shutdown.")



# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/CAM/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")




def get_public_ip():
    """
    Function to get the public IP address of the server.
    It uses an external service like ipinfo.io or ifconfig.co.
    """
    try:
        response = requests.get('https://ipinfo.io/ip')
        response.raise_for_status()
        public_ip = response.text.strip()  # Extract the public IP from the response
        local_logger.info(f"Detected public IP: {public_ip}")
        return public_ip
    except requests.RequestException as e:
        local_logger.error(f"Could not get public IP: {e}")
        return None

def setup_turn_server():
    try:
        config_path = '/etc/turnserver.conf'
        backup_path = '/etc/turnserver.conf.bak'

        # Fetch the public IP
        external_ip = get_public_ip()
        if not external_ip:
            local_logger.error("Failed to fetch the external IP. Aborting setup.")
            return

        # Backup existing configuration file if it exists
        if os.path.exists(config_path):
            shutil.copy(config_path, backup_path)
            local_logger.info(f"Existing configuration backed up to {backup_path}")

        # Generate a new configuration file for coturn
        with open(config_path, 'w') as config_file:
            config_file.write('listening-port=3478\n')
            config_file.write('tls-listening-port=5349\n')
            config_file.write('listening-ip=192.168.11.10\n')  # Internal IP
            config_file.write(f'external-ip={external_ip}\n')  # Use dynamically fetched public IP
            config_file.write('relay-ip=192.168.11.10\n')  # Internal IP
            config_file.write('min-port=10000\n')
            config_file.write('max-port=20000\n')
            config_file.write('user=BIH001:Brih$81\n')  # Authentication
            config_file.write('cli-password=Brih$81\n')
            config_file.write('fingerprint\n')
            config_file.write('cert=/home/bih01/CCTV/myssl/cert.pem\n')
            config_file.write('pkey=/home/bih01/CCTV/myssl/key.pem\n')
            config_file.write('dh-file=/home/bih01/CCTV/myssl/dhparam.pem\n')
            config_file.write('log-file=/var/log/turnserver/turn.log\n')
            config_file.write('snmp-config-file=/etc/snmp/turn.snmp.conf\n')
            config_file.write('userdb=/var/lib/turn/turndb\n')
            config_file.write('cli-ip=127.0.0.1\n')
            config_file.write('cli-port=5766\n')
        # Set the correct permissions for the configuration file
        os.chmod(config_path, 0o600)

        # Reload and restart coturn service
        sp.run(['systemctl', 'daemon-reload'])
        sp.run(['systemctl', 'restart', 'coturn'])
        local_logger.info("TURN server setup and started successfully.")

    except Exception as e:
        local_logger.error(f"An error occurred: {e}")

def check_and_setup_turn_server():
    try:
        # Check if coturn service is active
        result = sp.run(['systemctl', 'is-active', 'coturn'], capture_output=True, text=True)
        
        if result.stdout.strip() == 'active':
            local_logger.info("TURN server is already running")
            return
        
        # If not active, set up the server
        setup_turn_server()
    
    except Exception as e:
        local_logger.error(f"Error checking TURN server status: {e}")

# Call the function to check and set up the TURN server if needed
check_and_setup_turn_server()


# -----------------------------------------------------------------------------

def predict(model, img, classes=[], conf=0.5):
    """Predict with YOLO model."""
    results = model.predict(img, classes=classes if classes else None, conf=conf)
    return results

def object_detection(model, frame):
    # Run YOLO inference
    results = model(frame)
    if results and len(results) > 0:
        result = results[0]  # Get the first (and only) result
        return result.plot()  # Visualization
    return frame



class CameraCapture(VideoStreamTrack):
    def __init__(self, source, cam_name, from_file=None):
        super().__init__()
        self.from_file = from_file
        self.src = self.from_file if self.from_file is not None else source
        
        self.is_running = False
        self.stop_event = Event()
        self.t_cap = None

        # Initialize video capture
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source: {self.src}")

        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.FPS = int(self.cap.get(cv2.CAP_PROP_FPS))
        local_logger.debug(f"{cam_name} - Width: {self.width}, Height: {self.height}, FPS: {self.FPS}")

        # Initialize frame and timing
        self.frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)  # Placeholder frame
        self._pts = 0
        self._start_time = time.time()
        self._time_base = fractions.Fraction(1, self.FPS)

        # Camera name and recording variables
        self.cam_name = cam_name
        self.current_hour = None
        self.video_writer = None
        self.current_minute = -1

        # Object detection settings
        self.obj_detect = False
        self.resolution = 0.5  # Reduce resolution for faster processing



        # Initialize YOLO model and desired classes
        self.model = YOLO("yolo11n.pt")  # Use the smallest YOLO model for faster inference
        # Export the model to OpenVINO format
        self.model.export(format="openvino", half=True)  # Use FP32 for GPU

        # Load the exported OpenVINO model
        core = Core()
        self.ov_model = core.read_model("yolo11n_openvino_model/yolo11n.xml")
        self.compiled_model = core.compile_model(self.ov_model, "GPU")  # Use GPU

        self.desired_classes = ["car", "person", "cat", "dog", "truck", "motorcycle", "bicycle", "suitcase", "knife", "sport ball", "bird", "airplane", "mouse", "laptop", "keyboard", "remote"]
        self.class_indices = [idx for idx, name in self.model.names.items() if name.lower() in self.desired_classes]
        self.conf = 0.3  # Lower the confidence threshold for testing


    def preprocess_image(self, image, target_size=(640, 640)):
        """Preprocess the image for OpenVINO inference."""
        resized_image = cv2.resize(image, target_size)
        input_image = resized_image.transpose((2, 0, 1))  # HWC to CHW
        input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
        input_image = input_image.astype(np.float32)  # Convert to float32
        input_image /= 255.0  # Normalize pixel values
        return input_image

    async def predict_and_detect_iris(self, img):
        """Predict and draw detection results on the image."""
        print("Using OpenVINO model for inference")  # Debug statement

        # Preprocess the image
        input_image = self.preprocess_image(img)

        # Perform inference using OpenVINO
        results = self.compiled_model([input_image])[self.compiled_model.output(0)]

        # Debug: Print the shape and content of the results tensor
        print("Results shape:", results.shape)
        print("Results:", results)

        # Process results
        detections = self.process_results(results, confidence_threshold=self.conf)

        # Draw bounding boxes
        img_with_boxes = self.draw_boxes(img, detections, resolution=self.resolution)

        return img_with_boxes, detections

    def process_results(self, results, confidence_threshold=0.3):
        """Process OpenVINO output to extract detections."""
        detections = []
        output = results[0]  # Shape: [84, 8400]

        for i in range(output.shape[1]):
            row = output[:, i]

            # Extract confidence and class
            confidence = row[4:84].max()
            class_id = row[4:84].argmax()

            if confidence > confidence_threshold:
                # Extract bounding box coordinates
                x, y, w, h = row[:4]
                detections.append((
                    class_id,
                    confidence,
                    [x, y, w, h]  # Normalized to [0,1]
                ))

        print(f"OpenVINO Detections: {len(detections)}")  # Debug statement
        return detections

    def draw_boxes(self, image, detections, resolution=1.0):
        """Draw bounding boxes on the image."""
        for class_id, confidence, bbox in detections:
            center_x, center_y, width, height = bbox
            # Debug: Print bounding box coordinates in resized image space
            print(f"Resized Image BBox: center_x={center_x}, center_y={center_y}, width={width}, height={height}")

            # Convert center coordinates to corner coordinates
            x1 = center_x - (width / 2)
            y1 = center_y - (height / 2)
            x2 = center_x + (width / 2)
            y2 = center_y + (height / 2)

            # Scale bounding box coordinates to original image size
            scale_x = image.shape[1] / 640  # Scale factor for width
            scale_y = image.shape[0] / 640  # Scale factor for height

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            # Debug: Print scaled bounding box coordinates
            print(f"Scaled BBox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

            # Ensure coordinates are within the image bounds
            x1 = max(0, min(x1, image.shape[1] - 1))
            y1 = max(0, min(y1, image.shape[0] - 1))
            x2 = max(0, min(x2, image.shape[1] - 1))
            y2 = max(0, min(y2, image.shape[0] - 1))

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label and confidence
            label = f"{self.model.names[class_id]} {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return image


    async def predict_and_detect_cpu(self, img):
        """Predict and draw detection results on the image."""
        # Resize frame for faster processing
        # resized_img = cv2.resize(img, (int(self.width * self.resolution), int(self.height * self.resolution)))

        # Perform object detection
        results = self.model(img, classes=self.class_indices, conf=self.conf)

        # Process results and draw bounding boxes
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0]
                cls = int(box.cls[0])
                conf_score = box.conf[0] * 100  # Confidence as a percentage
                label = result.names[cls]
                if "airplane" in label and self.conf < 0.5:
                    label = "car"
                label = f"{label} : {conf_score:.0f}%"  # Add confidence to the label

                # Scale bounding box coordinates back to original resolution
                xyxy = [int(coord / self.resolution) for coord in xyxy]

                # Draw bounding box
                pos1 = (int(xyxy[0]), int(xyxy[1]) - 10)
                pos2 = (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(img, pos1, pos2, (0, 255, 0), 2)

                # Draw label background
                text_size, _ = cv2.getTextSize(label.capitalize(), cv2.FONT_HERSHEY_PLAIN, 2, 2)
                text_width, text_height = text_size
                text_bg_top_left = (pos1[0] - 1, pos1[1] - text_height - 8)
                text_bg_bottom_right = (pos1[0] + text_width, pos1[1])
                cv2.rectangle(img, text_bg_top_left, text_bg_bottom_right, (0, 255, 0), -1)

                # Add label
                text_position = (pos1[0], pos1[1] - 5)
                cv2.putText(img, label.capitalize(), text_position, cv2.FONT_HERSHEY_PLAIN, 2, (200, 10, 245), 2, cv2.LINE_AA)

        return img, results



    async def recv(self):
        try:
            if not self.is_running:
                raise ValueError('Cemera Stopped.')
            
            # # results = self.model.track(self.frame, persist=True)
            # results = self.model.track(source=self.frame, tracker="/home/bih01/webrtc_opencv_fastapi/dataset/custom_dataset.yaml")

            # # Visualize the results on the frame
            # annotated_frame = results[0].plot()


            video_frame = VideoFrame.from_ndarray(self.frame, format='bgr24')
            video_frame.pts = self._pts
            video_frame.time_base = self._time_base

            return video_frame

        except Exception as e:
            local_logger.error(f"Error in recv method:\n{str(traceback.format_exc())}")
            video_frame.pts, video_frame.time_base = await self.next_timestamp()
            return

    def get_output_filename(self, cam_name):
        # Generate a filename with the current date and hour
        rec_dir = f"/mnt/sdb1/REC_{cam_name}"
        now = datetime.now()
        os.makedirs(rec_dir, exist_ok=True)

        return f"{rec_dir}/{cam_name}_{now.strftime('%Y-%m-%d_%H-%M')}.mp4"

    def create_video_writer(self, cam_name, fps):
        # Initialize a new VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        filename = self.get_output_filename(cam_name)
        return cv2.VideoWriter(filename, fourcc, fps, (self.width, self.height))


    async def perspective_transform(self, img, src_points, dst_points, output_size):

        src_points = np.float32(src_points)
        dst_points = np.float32(dst_points)

        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply perspective transformation
        result = cv2.warpPerspective(img, matrix, (img.shape[1],img.shape[0]))

        return result



    async def _run(self):
        prev_pt = int((time.time() - self._start_time) / self._time_base)
        t = time.time()
        fps = 0.0


        roi = {'camera1': [20, -30, 260, -1], 
               'camera2': [200, -200, 1, -1], 
               'camera3': [350, -500, 100, -1],
               'camera4': [1, -1, 1, -1]}
        
        y0, y1 = roi[self.cam_name][:2]
        x0, x1 = roi[self.cam_name][2:]

        src_points = [
            [0,    200],   # P1
            [0,    720],   # P2
            [1280, 400],   # P3
            [1280, 720]    # P4
        ]

        dst_points = [
            [0,    200],   # P1'
            [0,    720],   # P2'
            [1280, 320],   # P3'
            [1280, 720]    # P4'
        ]

        
        local_logger.info('---Processing Camera...')


        while self.cap.isOpened():
            if self.stop_event.is_set():
                local_logger.warning(f"Event is set to stop for  {self.cam_name}")
                break
            try:
                success, image = await asyncio.to_thread(self.cap.read)

                if not success:
                    local_logger.warning(f"Failed to read frame from {self.cam_name}")
                    await asyncio.sleep(3)
                    continue
                
                if '2' in self.cam_name:
                    # Apply transformation
                    image = await self.perspective_transform(image, src_points, dst_points, (self.width, self.height))

                processed_frame = image#[y0:y1, x0:x1]

                if self.obj_detect:
                    processed_frame, _ = await self.predict_and_detect_cpu(processed_frame)

                resized = cv2.resize(processed_frame, None, fx=self.resolution, fy=self.resolution, interpolation=cv2.INTER_LINEAR)
                # resized_frame[:40, :] = [0,0,0]
                self.frame = cv2.putText(resized, f'{datetime.now():%T} FPS: {fps:.1f}', (3, resized.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
                
                if time.time() > t+5:
                    fps = (self._pts - prev_pt) / 5
                    prev_pt = self._pts
                    t = time.time()
                self._pts = int((time.time() - self._start_time) / self._time_base)

                #await asyncio.sleep(0.001)

                now = datetime.now()
                current_minute = now.minute
                if self.current_minute != current_minute // 1:
                    # Hour has changed, start a new video file
                    local_logger.info("Hour has changed, start a new video file")
                    if self.video_writer:
                        self.video_writer.release()  # Release the current writer
                    self.video_writer = self.create_video_writer(self.cam_name, self.FPS)
                    self.current_minute = current_minute // 1
                frame_with_text = processed_frame.copy()

                cv2.putText(
                    frame_with_text,
                    f'{datetime.now():%T} FPS: {fps:.1f}, time_base: {self._time_base}',
                    (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
                
                await asyncio.to_thread(self.video_writer.write, frame_with_text)

            except Exception as e:
                local_logger.error(f"Camera ERROR: \n{traceback.format_exc()}")




    def run(self):
        asyncio.run(self._run())
 
    def start_capture(self):
        """
        Start capturing video frames.
        """
        if self.is_running:
            local_logger.info("Capture is already running")
            return
        if not self.cap.isOpened():
            local_logger.error(f"Failed to open video source")
            return
        self.t_cap = Thread(target=self.run)
        self.t_cap.start()
        self.is_running = True
        local_logger.info("Camera capture started!")

    def stop_capture(self):
        """
        Stop capturing video frames.
        """
        if not self.is_running:
            local_logger.info("Capture is not running")
            return
        local_logger.info("Stopping the camera capture thread")
        self.stop_event.set()
        self.is_running = False

        # Wait for the thread to finish
        if self.t_cap and self.t_cap.is_alive():
            self.t_cap.join()  # Gracefully wait for the thread to finish
            local_logger.info("Camera capture thread stopped.")

        # Release resources
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
                
    async def process_frame(self, frame):
        """
        Process a single frame for object detection and return the updated frame.
        :param frame: The image frame to be processed
        :return: The processed image frame with detections
        """
        return cv2.GaussianBlur(frame,(5,5),0)



# local_logger.info(f"\r\n{sd.query_devices(device=0, kind='input')}\r\n")
# local_logger.info(f"\r\n{sd.query_devices()}\r\n")




try:
    camera1_capture = CameraCapture(CAMERA_URLS["camera4"], cam_name='camera4')
    camera1_capture.start_capture()
    CAMERA_CAP["camera1"] = camera1_capture
except Exception as e:
    local_logger.error(e)


# v4l2-ctl --list-devices


# Use device index found from 'arecord -l'
# from audioStreamTrack import AudioStreamTrack
# audio_capture = AudioStreamTrack(logger=local_logger)  
##audio_capture.start()

@app.websocket("/CAM/bufferSize")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # client = websocket.headers.get("X-Forwarded-For", websocket.headers.get("X-Real-IP"), websocket.headers.get("host"))
    local_logger.info(f"Connection Accepted from client: {1}")
    
    try:
        while True:
            data = await websocket.receive_json()

            if data['type'] == 'buffer_size':
                new_buffer_size = data['value']
                local_logger.info(f"Received new buffer size: {new_buffer_size}")
                
                # Update the buffer size in the audio stream
                # audio_capture.update_buffer_size(freq=new_buffer_size)
    except Exception as e:
        local_logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# Track peer connections
pcs = set()

from pcManager import PCManager

data = {'restart_service': 1, 
        'camera1': {'detect_object': 0, 'resize': [.75, .75], 'operation': 'on'},
        'camera2': {'detect_object': 0, 'resize': [.75, .75], 'operation': 'on'},
        'camera3': {'detect_object': 0, 'resize': [.75, .75], 'operation': 'on'},
        'camera4': {'detect_object': 0, 'resize': [1.0, 1.0], 'operation': 'on'}
        }

async def restart_service():
    try:
        # Use asyncio to asynchronously run the systemctl restart command
        process = await asyncio.create_subprocess_exec(
            "systemctl", "restart", "brih-stream",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for the process to complete
        stdout, stderr = await process.communicate()

        # Check for errors in the process
        if process.returncode != 0:
            print(f"Failed to restart service: {stderr.decode()}")
        else:
            print(f"Service restarted successfully: {stdout.decode()}")
    except Exception as e:
        print(f"Error restarting service: {str(e)}")





@app.post("/CAM/api")
async def api(request: Request):
    # Send immediate feedback to the frontend
    data = await request.json()
    local_logger.info(json.dumps(data, indent=2))

    
    targetObject = CAMERA_CAP.get(data.get("camera_name", None), None)
    
    if targetObject is not None:
        if data['cameraOperation'] == 0:
            targetObject.stop_capture()
            return {'response': 'Operation Success!'}
        
        if  data['cameraOperation'] == 1:
            CAMERA_CAP[data["camera_name"]] = CameraCapture(CAMERA_URLS[data["camera_name"]], cam_name=data["camera_name"])
            return {'response': 'Operation Success!'}
        
        targetObject.resolution = data['resolution'] / 100
        targetObject.obj_detect = data['objectDetect']

        output_file = f'{os.getcwd()}/{data["camera_name"]}.json'

        with open(output_file, "w") as file:
            json.dump(data, file, indent=4)
        return {'response': 'Operation Success!'}

    if data.get('restart_service', None) is not None:
        response = {"status": "Restarting service... Please wait."}

        # Start the system restart asynchronously without blocking the response
        asyncio.create_task(restart_service())

        return response
    
    return {'response': 'Operation Failed!'}

@app.post("/xray/api")
async def api(request: Request):
    # Send immediate feedback to the frontend
    data = await request.json()
    local_logger.info(json.dumps(data, indent=2))
    
    if data.get('service', None) is not None:
        command = data['service']
        response = {"status": f"Processing {command} command, Please wait..."}

        asyncio.create_task(process_service())

        return response
    
    return {'response': 'Operation Failed!'}


async def process_service(command):
    try:
        # Use asyncio to asynchronously run the systemctl restart command
        process = await asyncio.create_subprocess_exec(
            "systemctl", command, "xray-service",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for the process to complete
        stdout, stderr = await process.communicate()

        # Check for errors in the process
        if process.returncode != 0:
            print(f"Failed to restart service: {stderr.decode()}")
        else:
            print(f"Service restarted successfully: {stdout.decode()}")
    except Exception as e:
        print(f"Error restarting service: {str(e)}")

@app.get("/xray/service-status")
async def service_status(request: Request):
    async def status_generator():
        try:
            local_logger.info('Checking service status...')
            process = await asyncio.create_subprocess_exec(
                "systemctl", "is-active", "--quiet", "brih-stream",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                status = "Service is running."
            else:
                status = "Service is not running or failed to start."
                
        except Exception as e:
            status = f"Error checking service status: {str(e)}"

        yield f"data: {json.dumps({'status': status})}\n\n"
        await asyncio.sleep(10)

    return StreamingResponse(
        status_generator(),
        media_type="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'text/event-stream'
        }
    )

@app.post("/CAM/offer")
async def offer(request: Request):
    client = request.headers.get("X-Forwarded-For", request.headers.get("X-Real-IP", request.client.host))
    data = await request.json()
    offer_data = data["offer_data"]
    cameraObject = data["camera_id"]
    local_logger.info(f'\n\ncamera ID : {cameraObject}')

    if not cameraObject:
        return

    try:
        config = RTCConfiguration(
                    iceServers=[
                        # Free public STUN servers
                        # RTCIceServer(urls="stun:stun.l.google.com:19302"),
                        # RTCIceServer(urls="stun:stun1.l.google.com:19302"),
                        # RTCIceServer(urls="stun:stun2.l.google.com:19302"),
                        # RTCIceServer(urls="stun:stun3.l.google.com:19302"),
                        # RTCIceServer(urls="stun:stun4.l.google.com:19302"),

                        RTCIceServer(
                            urls="turn:0.0.0.0:3478",
                            username="BIH001",
                            credential="Brih$81"
                        ),
                    ],
        )

        config.iceCandidatePoolSize = 10

        # local_logger.debug(f"RTCConfiguration: {config.__dict__}")
        pc = RTCPeerConnection(configuration=config)

        
        # @pc.on("connectionstatechange")
        # async def on_connectionstatechange():
        #     local_logger.info(f"Connection state changed to {pc.connectionState}")
        #     if pc.connectionState != 'connected':
        #         for name, camObj in CAMERA_CAP.items():
        #             local_logger.info(f"\nStarting camera {name}.....")
        #             try:
        #                 camObj.start_capture()
        #             except Exception as e:
        #                 local_logger.error(f'ERROR with  {name}, {e}')
                    
        #     else:
        #         for name, camObj in CAMERA_CAP.items():
        #             if name != cameraObject:
        #                 local_logger.info(f"\nStopping {name} and Focusing only on {cameraObject}")
        #                 camObj.stop_capture()



        # Initialize your PCManager and add tracks
        pc_manager = PCManager(pc, config, local_logger, pcs)

        # Add tracks (video and audio) to the peer connection
        pc_manager.add_tracks(CAMERA_CAP[cameraObject])#, audio_capture)

        # Handle the offer and set remote description
        await pc_manager.set_remote_description(offer_data)

        # Create and set the local description (answer)
        answer = await pc_manager.create_answer()

        # Log connection info
        pc_manager.log_connection_info()

        local_logger.debug("Offer handled successfully")
        local_logger.info(f'\r\n{"-"*100}\r\n')
        return JSONResponse({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    except Exception as e:
        local_logger.error(f"Error handling offer: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse({"error": "Failed to handle offer"}, status_code=500)
    

@app.get("/CAM/service-status")
async def service_status():
    async def status_generator():
        try:
            local_logger.info('Checking service status...')
            process = await asyncio.create_subprocess_exec(
                "systemctl", "is-active", "--quiet", "brih-stream",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                status = "Service is running."
            else:
                status = "Service is not running or failed to start."
                
        except Exception as e:
            status = f"Error checking service status: {str(e)}"

        yield f"data: {json.dumps({'status': status})}\n\n"
        await asyncio.sleep(10)

    return StreamingResponse(
        status_generator(),
        media_type="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'text/event-stream'
        }
    )


@app.get("/CAM/test", response_class=HTMLResponse)
async def test(request: Request):
    return templates.TemplateResponse("test.html", {"request": request})


@app.get("/CAM/index", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/CAM/cameras", response_class=HTMLResponse)
async def cameras(request: Request):
    return templates.TemplateResponse("cameras.html", {"request": request})



@app.get("/BrIH/majid/cctv/v1/auth/login", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/CAM/cv", response_class=HTMLResponse)
async def cv(request: Request):
    local_logger.info("Return template")
    return templates.TemplateResponse("index_cv.html", {"request": request, "username": "Me"})

@app.get("/CAM/get-index", response_class=HTMLResponse)
async def serve_index(request: Request):
    file = os.path.join(os.getcwd(), 'templates', 'client_index.html')
    return FileResponse(file)




# -------------------------------------------------------------------------------


# from clone import disk_clone


# Thread(target=disk_clone).start()
        
# @app.middleware("http")
# async def add_client_ip(request: Request, call_next):
#     client_ip = request.headers.get("X-Real-IP") or request.client.host
#     local_logger.info(f'add_client_ip: {client_ip}.')

#     response = await call_next(request)
#     response.headers["X-Client-IP"] = client_ip
#     return response

from starlette.middleware.base import BaseHTTPMiddleware



# ------------------------------------------------------------------------------------------------------------------------------

@app.get("/CAM/get_template/{filename}")
def download_html_file(request: Request, filename: str):
    # Construct the file path correctly
    HTML_FILE_PATH = os.path.join(os.getcwd(), 'templates', 'clientTemplates', filename)
    local_logger.info(HTML_FILE_PATH)
    
    # Ensure the file exists
    if not os.path.exists(HTML_FILE_PATH):
        return {"error": "File not found."}

    # Return the file as a response
    return FileResponse(
        path=HTML_FILE_PATH,
        media_type="text/html",
        filename=filename, 
    )




security = HTTPBasic()



# Secret key for JWT - in production, use a secure environment variable
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"

# Mock database - in production, use a real database
USERS_DB = {
    "admin": {
        "username": "admin",
        "password": bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode(),
        "role": "admin"
    }
}

class LoginData(BaseModel):
    username: str
    password: str
    remember: Optional[bool] = False


@app.get("/CAM/user", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(os.path.join("clientTemplates", "index.html"), {"request": request})

@app.post("/CAM/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    remember: bool = Form(False)
):
    
    # Verify credentials
    user = USERS_DB.get(username)
    local_logger.info(f"user: {user}")
    if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
        return JSONResponse(
            content={"redirected": True, "url": "http://0.0.0.0:8000/login?error=Invalid credentials, Try again!"},
            status_code=400
        )


    # Redirect URL (after successful login)
    redirect_url = await redirectUser(username)
    local_logger.info(f"Redirecting to: {redirect_url}")
    
    # Set cookie in response
    response = JSONResponse(content={"redirected": True, "url": redirect_url})

    return response


@app.post("/CAM/login--")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    remember: bool = Form(False)
):
    local_logger.info("Received login request")
    local_logger.info(f"Username: {username}, Remember me: {remember}")

    # Verify credentials
    user = USERS_DB.get(username)
    if not user:
        local_logger.warning(f"User '{username}' not found in the database.")
        return RedirectResponse(
            "http://0.0.0.0:8000/login?error=Invalid credentials, Try again!",
            status_code=status.HTTP_302_FOUND,
            headers={"x-error": "Invalid credentials"},
        )
    
    if not bcrypt.checkpw(password.encode(), user["password"].encode()):
        local_logger.warning(f"Password mismatch for user '{username}'.")
        return RedirectResponse(
            "http://0.0.0.0:8000/login?error=Invalid credentials, Try again!",
            status_code=status.HTTP_302_FOUND,
            headers={"x-error": "Invalid credentials"},
        )

    local_logger.info(f"User '{username}' authenticated successfully.")

    # Generate JWT token
    token_expiry = datetime.now() + timedelta(days=30 if remember else 1)
    token_data = {
        "sub": username,
        "role": user["role"],
        "exp": token_expiry.timestamp(),  # Unix timestamp
    }
    local_logger.info(f"Token data: {token_data}")

    try:
        token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
        local_logger.info("JWT token generated successfully.")
    except Exception as e:
        local_logger.error(f"Failed to generate JWT token: {e}")
        raise

    local_logger.info("JWT token generated successfully.")

    # Redirect after successful login and set cookie
    try:
        redirect_url = await redirectUser(username)
        local_logger.info(f"Redirect URL: {redirect_url}")
        response = RedirectResponse(url=redirect_url, status_code=303)
        response.set_cookie(
            key="access_token",
            value=f"Bearer {token}",
            httponly=True,
            secure=True,  # Secure flag is necessary for SameSite=None
            samesite="None",  # Cross-site requests
            expires=token_expiry.timestamp()
        )

        local_logger.info("Cookie set successfully.")
    except Exception as e:
        local_logger.error(f"Failed during redirection or setting cookie: {e}")
        raise

    local_logger.info(f"Redirecting to {redirect_url}")
    return response


@app.get("/CAM/logout")
async def logout():
    response = RedirectResponse(url="http://0.0.0.0:8000/login", status_code=303)
    response.delete_cookie("access_token")
    return response

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    if exc.status_code == 401:

        return RedirectResponse("http://0.0.0.0:8000/login?error=Invalid credentials, Try again!", status_code=status.HTTP_302_FOUND, headers={"x-error": "Invalid credentials"})
    return exc


@app.get("/CAM/public")
async def whatismyip(request: Request):
    client = client = request.headers.get("X-Forwarded-For")
    local_logger.info(client)
    return {"ip": client.split(',')[0]}


async def redirectUser(username):
    local_logger.info(f"Redirecting user: {username}")
    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.get("https://cheaply-verified-squid.ngrok-free.app/CAM/public")
            response.raise_for_status() 
            addr = response.json()
        except httpx.RequestError as exc:
            local_logger.error(f"An error occurred while requesting: {exc}")
            return {"error": "Failed to fetch external IP address."}
        except httpx.HTTPStatusError as exc:
            local_logger.error(f"HTTP error occurred: {exc}")
            return {"error": "Invalid response from the external service."}

    if 'ip' not in addr:
        local_logger.error(f"error: Invalid response format from the external service.")
        return {"error": "Invalid response format from the external service."}

    redirect_url = f"https://{addr['ip']}/CAM/cameras"
    
    return redirect_url




@app.get("/CAM/error_page", response_class=HTMLResponse)
async def error_page(request: Request):
    return templates.TemplateResponse("clientTemplates/error_page_js.html", {"request": request})




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any origin
    allow_credentials=False,  # Do not allow credentials (cookies, headers)
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)



if __name__ == "__main__":
    # import uvicorn
    # uvicorn.run("main:app", host="0.0.0.0", port=2230)
    from hypercorn.config import Config
    from hypercorn.asyncio import serve
    addr = "127.0.0.1"

    config = Config()

    http = [f'{addr}:{port}' for port in range(2230, 2240)]
    # SSL certificate and key
    # config.certfile = "/home/bih01/Desktop/BIH-GPT/ca.crt"
    # config.keyfile = "/home/bih01/Desktop/BIH-GPT/ca.key"
    # config.ssl = True  # Enable SSL
    # http.append("192.168.11.10:7777")

    # https = [f'{server_addr}:443', f'{addr}:443']
    config.bind = http


    asyncio.run(serve(app, config)) 
