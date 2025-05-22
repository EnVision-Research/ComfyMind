"""
ComfyUI Tools Configuration and Implementation
"""
import logging
logger = logging.getLogger(__name__) 

import os
import uuid
import json
import logging
import random
import urllib.parse
import urllib.request
import websocket
import base64
import requests
from typing import Dict, Optional, Any, List
from pydantic import BaseModel
import re
from datetime import datetime
from pathlib import Path
from PIL import Image
import io
import cv2 
from openai import OpenAI
import yaml

# ============= Configuration =============
with open('./config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

    COMPLETION_BASE_URL = config['completion']['base_url']
    COMPLETION_API_KEY = config['completion']['api_key']
    COMPLETION_MODEL_NAME = config['completion']['model_name']
    COMPLETION_HYPER_PARAMETER = config['completion']['hyper_parameter']

    VISION_BASE_URL = config['vision']['base_url']
    VISION_API_KEY = config['vision']['api_key']
    VISION_MODEL_NAME = config['vision']['model_name']
    VISION_HYPER_PARAMETER = config['vision']['hyper_parameter']

    VISION_REASONING_BASE_URL = config['vision_reasoning']['base_url']
    VISION_REASONING_API_KEY = config['vision_reasoning']['api_key']
    VISION_REASONING_MODEL_NAME = config['vision_reasoning']['model_name']

    TEXT_REASONING_BASE_URL = config['text_reasoning']['base_url']
    TEXT_REASONING_API_KEY = config['text_reasoning']['api_key']
    TEXT_REASONING_MODEL_NAME = config['text_reasoning']['model_name']

    COMFYUI_SERVER = config['comfyui']['base_url']
    COMFYUI_WS_SERVER = config['comfyui']['ws_base_url']
    COMFYUI_OUTPUT_DIR = config['comfyui']['output_dir']
    COMFYUI_WORKFLOW_DIR = config['comfyui']['workflow_dir']
    COMFYUI_WORKFLOW_META_INFO_PATH = config['comfyui']['workflow_meta_info_path']

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class ComfyUITools:
    def __init__(self):
        self.base_url = COMFYUI_SERVER
        self.ws_base_url = COMFYUI_WS_SERVER
        self.default_headers = {"Content-Type": "application/json"}

        self.output_dir = COMFYUI_OUTPUT_DIR
        self.workflow_folder_path = COMFYUI_WORKFLOW_DIR
        self.Workflow_meta_info_path = COMFYUI_WORKFLOW_META_INFO_PATH

        with open(self.Workflow_meta_info_path, 'r', encoding='utf-8') as f:
            self.Workflow_meta_info = json.load(f)

    def queue_prompt(self, prompt: dict, client_id: str) -> dict:
        """Queue a prompt for processing."""
        data = json.dumps({"prompt": prompt, "client_id": client_id}).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/prompt", data=data, headers=self.default_headers)
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())

    def get_history(self, prompt_id: str) -> dict:
        """Get the history for a prompt."""
        req = urllib.request.Request(f"{self.base_url}/history/{prompt_id}", headers=self.default_headers)
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())

    def get_image_url(self, filename: str, subfolder: str, folder_type: str) -> str:
        """Generate the URL for an image."""
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        return f"{self.base_url}/view?{url_values}"

    def load_workflow_file(self, workflow_name: str) -> dict:
        """Load workflow from filename

        Args:
            workflow_name: The name of the workflow file
        Returns:
            Loaded workflow dictionary or error message string
        """
        try:
            workflow_path = os.path.join(
                self.workflow_folder_path, workflow_name
            )
            try:
                with open(workflow_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except FileNotFoundError:
                logging.error(f"Error: Workflow file not found. File path: {workflow_path}")
                return {"error": "Workflow file not found."}
            except json.JSONDecodeError:
                logging.error(f"Error: Invalid JSON in workflow file: {workflow_path}")
                return {"error": "Invalid JSON in workflow file."}
            except IOError as e:
                logging.error(f"Error: Failed to read workflow file: {str(e)}")
                return {"error": "Failed to read workflow file."}

        except Exception as e:
            logging.error(f"Error loading workflow file: {str(e)}")
            return {"error": "Error loading workflow file."}

    def invoke_vision(self, message: any, model_name = None, api_key = None, base_url = None) -> tuple[str, any]:

        if base_url and api_key:
            client = OpenAI(
                base_url=base_url,
                api_key=api_key
            )
        else:
            client = OpenAI(
                base_url=VISION_BASE_URL,
                api_key=VISION_API_KEY
            )

        try:
            if model_name is None:
                response = client.chat.completions.create(
                    model=VISION_MODEL_NAME,
                    messages=message,
                    **VISION_HYPER_PARAMETER
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=message,
                    **VISION_HYPER_PARAMETER
                )
            answer = response.choices[0].message.content
            usage = response.usage

        except Exception as error:
            answer = f'Error: {error}'
            usage = None

        return answer, usage

    def invoke_vision_reasoning(self, message: any, model_name = None, api_key = None, base_url = None) -> tuple[str, any]:
        if base_url and api_key:
            client = OpenAI(
                base_url=base_url,
                api_key=api_key
            )
        else:
            client = OpenAI(
                base_url=VISION_REASONING_BASE_URL,
                api_key=VISION_REASONING_API_KEY
            )

        try:
            print("Use Model: ", TEXT_REASONING_MODEL_NAME)
            response = client.chat.completions.create(
                model=VISION_REASONING_MODEL_NAME,
                messages=message
            )
            answer = response.choices[0].message.content
            usage = response.usage

        except Exception as error:
            answer = f'Error: {error}'
            usage = None

        return answer, usage

    def save_and_get_image(self, filename: str, subfolder: str, folder_type: str) -> str:
        """Save image to local cache and return relative URL path."""
        try:
            # Get image from ComfyUI
            url = self.get_image_url(filename, subfolder, folder_type)
            response = requests.get(url)
            response.raise_for_status()

            # Prepare cache directory
            cache_dir = self.output_dir
            os.makedirs(cache_dir, exist_ok=True)

            # Save image with original filename
            cache_path = os.path.join(cache_dir, filename)
            with open(cache_path, "wb") as f:
                f.write(response.content)

            # Return relative URL path
            return cache_path

        except Exception as e:
            logging.error(f"Error saving image: {str(e)}")
            return None

    def encode_video_to_base64(self, video_path: str) -> list:
        base64_frames = []
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            status, frame = video.read()
            if not status:
                break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            base64_frame = base64.b64encode(buffer.getvalue()).decode('utf-8')
            base64_frames.append(base64_frame)
        video.release()
        return base64_frames

    def extract_json(self, content: str) -> dict:  
        json_pattern = r'```json(.*?)```'  
        json_matches = re.findall(json_pattern, content, re.DOTALL)  

        for match in json_matches:  
            try:  
                parsed_json = json.loads(match.strip())  
                return parsed_json  
            except json.JSONDecodeError:  
                continue 

        return None  

    def fit_requirements(self, workflow, prompt, input_images, input_videos, other_parameters, requirements_message):
        """
        Using VLM to fit the requirements to the workflow by editing the parameters but not change the nodes and its connections.
        """
        system_prompt = """
        You are an advanced AI model tasked with customizing workflows based on specific user requirements and visual inputs. Your primary goal is to adjust hyperparameters without altering the workflow's structure or node connections. Follow these steps to achieve this:
        1. Analyze Visual Input:
        - Carefully examine the provided visual content to understand its key elements and context.
        - Identify any specific features or areas relevant to the user's request (e.g., determining where to place an object like a small dog).
        2. Understand User Requirements:
        - Clearly interpret the user's specific demands, such as adding an object to an image or modifying certain aspects of the workflow.
        - Ensure you fully grasp the requirements before proceeding with any changes.
        - Ignore requests about generation quality. For example: high quality, seamless integration,  without visible artifacts, etc.
        3. Review Workflow Structure:
        - Analyze the given workflow to understand its current structure and node connections.
        - Identify which hyperparameters control the aspects you need to modify based on the user's request (e.g. The length of the video is controled by video frames and fps)(e.g. Analyze which nodes control which prompt's layout).
        - Think step by step how to modify the parameters. (e.g. Video length(seconds) = video frames / fps(frame_rate), so you should add or reduce the video frames to make video frames / fps(frame_rate) == the required length)(e.g. Modify the layout parameters of the corresponding prompt to meet the requirements))
        4. Modify Hyperparameters:
        - Adjust only the necessary hyperparameters to meet the user's requirements, ensuring the workflow's structure and node connections remain unchanged.
        - Base your modifications on the analysis of the visual input and the user's specific demands.
        5. Provide Clear Output:
        - Return the modified workflow with a clear explanation of the changes made.
        - Ensure the output is easy to understand and directly addresses the user's needs.
        Very Important Attention:
        - The output *must* follow the format:
            1.Chain of Thought 2.```json<modified workflow>```
        - The video duration error does not need to be modified if it is within 0.5s
        - The number of video frames of the video generation model *WAN* cannot be selected arbitrarily, it must meet the condition modulo 4 and remainder 1. For example (25, 29, 33 ...)
        Limitation:
        - *DO NOT* change any prompt, and any path of the files.
        - Must follow the rule of JSON format. Do not add any other comment.
        Example (Chain of Thought):
        Suppose the requirement is to add a small dog to an image. First, confirm the image resolution to ensure compatibility with the workflow. Next, analyze the given image to determine the most suitable location for placing the small dog, considering factors like existing objects and composition. Then, review the provided workflow to identify which hyperparameters control the insertion coordinates of the small dog. Based on your analysis, modify these hyperparameters to place the dog appropriately. Finally, return the updated workflow with a clear explanation of the changes made. 
        Next, I will provide you with the image/video, workflow and the requirements. Think step by step.
        """
        content = []
        meta_info = {}

        if input_images:
            for key, value in input_images.items():
                image = Image.open(value)  
                buffer = io.BytesIO()  
                image.save(buffer, format="PNG")  
                result_base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                meta_info[value] = image.size
                content.append({  
                    "type": "image_url",  
                    "image_url": {  
                        "url": f"data:image/png;base64,{result_base64_image}"  # Correct MIME type for PNG  
                    }  
                })  
        if input_videos:
            for key, value in input_videos.items():
                video_meta = self.get_video_metadata(value)  # Get video metadata
                meta_info[value] = video_meta

        content.append({
            'type': 'text',  
            'text': f"The requirements are: {requirements_message}. And the workflow is: {workflow}. And the meta info of the input images/videos are: {meta_info}. The information of the workflow is {self.Workflow_meta_info}."  
        })
        message = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content
            }    
        ]  
        response, usage = self.invoke_vision(message)

        # Print the thinking part
        logger.info(f"Response: {response}")
        print(f"Response: {response}")
        workflow = self.extract_json(response)
        # print(f"Modified workflow: {workflow}")
        return workflow

    def get_video_metadata(self, video_path: str) -> dict:
        """Extract metadata from the video."""
        video = cv2.VideoCapture(video_path)
        meta_info = {
            'width': int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'num_frames': int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
            'frame_rate': int(video.get(cv2.CAP_PROP_FPS)),
            'duration': int(video.get(cv2.CAP_PROP_FRAME_COUNT)) / int(video.get(cv2.CAP_PROP_FPS))
        }
        video.release()
        return meta_info

    def upscale_image(self, image_path: str) -> str:
        workflow_for_upscale = self.load_workflow_file('009_api.json')
        workflow_str = json.dumps(workflow_for_upscale)
        workflow_str = workflow_str.replace('%%IMAGE1%%', image_path)
        workflow_for_upscale = json.loads(workflow_str)

        # Upscale the image
        client_id = str(uuid.uuid4())  
        output_files = []  
        try:  
            ws_url = f"{self.ws_base_url}/ws?clientId={client_id}"  
            ws = websocket.WebSocket()  
            ws.connect(ws_url)  
            prompt_id = self.queue_prompt(workflow_for_upscale, client_id)["prompt_id"]  
            print("Start to wait for completion".center(80, '-'))
            while True:  
                out = ws.recv()  
                if isinstance(out, str):  
                    message = json.loads(out)  
                    if message["type"] == "executing":  
                        data = message["data"]  
                        if data["node"] is None and data["prompt_id"] == prompt_id:  
                            break  
            print("End wait for completion".center(80, '-'))
            history = self.get_history(prompt_id)[prompt_id]  
            for node_id, node_output in history["outputs"].items():    
                if "images" in node_output:  
                    for image in node_output["images"]:  
                        image_path = self.save_and_get_image(  
                            image["filename"], image["subfolder"], image["type"]  
                        )  
                        if image_path:  
                            output_files.append({"images": image_path})  

            print(f"Upscaled image: {output_files[0]['images']}")
            return output_files[0]["images"]
         
        except Exception as e:  
            logging.error(f"Error in generate_using_workflow: {str(e)}")  
            return {"error": str(e)}  
        finally:  
            if "ws" in locals():  
                ws.close()

    def keep_largest_connected_component_color(self, mask_color):
        """
        Keep the largest connected component in the mask
        """
        if len(mask_color.shape) != 3 or mask_color.shape[2] != 3:
            raise ValueError("The input image must be a color three-channel image")

        # Convert to grayscale and extract connected components
        gray = cv2.cvtColor(mask_color, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Connected component analysis
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        if num_labels <= 1:
            return mask_color  # No white area

        import numpy as np
        # Find the largest component (skip background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        # Construct the largest area mask
        largest_mask = (labels == largest_label).astype(np.uint8) * 255
        # Create a color output image, only keep the largest white block
        output = np.zeros_like(mask_color)
        output[largest_mask == 255] = [255, 255, 255]

        return output


    def generate_using_workflow_reason_inpainting(self,   
                                workflow_name: str,  # Make workflow_name a required parameter
                                requirements_message: Optional[str] = None,
                                prompt: Optional[Dict[str, str]] = None,   
                                input_images: Optional[Dict[str, str]] = None,   
                                input_videos: Optional[Dict[str, str]] = None,   
                                other_parameters: Optional[Dict[str, str]] = None) -> Dict:  
        # Step 1: Image Resize
        # Step 2: Use VLM to reasoning the editing area, get two points: Left Top and Right Bottom, and get mask-prompt. Then the image part in box is as image' and will be add a mask
        # Step 3: Mask the image' according to the mask-prompt and convert into a mask of image
        # Step 4: Using Inpainting workflow to generate the new image

        # prepare
        instruction = None
        for key, value in prompt.items():
            if key == '%%PROMPT1%%':
                instruction = value

        # If min of width and height of the image is less than 768, then upscale the image by 2x
        for key, value in input_images.items():
            image = Image.open(value)
            if min(image.size) <= 768:
                upscaled_image_path = self.upscale_image(value) # temporary save the upscaled image
                image = Image.open(upscaled_image_path)
                image.save(value.replace('.png', '_upscaled.png')) # save the upscaled image
                input_images[key] = value.replace('.png', '_upscaled.png')
                logger.info(f"Upscaled image has been saved to {input_images[key]}")


        logger.info(f"Start to resize the image to 512 * 512")
        # Step 1: Image Resize
        resized_length = 512
        ori_image_path = None
        for key, value in input_images.items():
            image = Image.open(value)
            ori_image = Image.open(value)
            ori_image_path = value
            image = image.resize((resized_length, resized_length))
            # save as another image(dont change the original image), save the new image path as image_resized
            image.save(value.replace('.png', '_resized.png'))
            image_resized_path = value.replace('.png', '_resized.png')

        ori_image_width = ori_image.size[0]
        ori_image_height = ori_image.size[1]
        logger.info(f"Resized image path: {image_resized_path}")

        logger.info(f"Start to use VLM to reasoning the editing area, get two points: Left Top and Right Bottom, and get mask-prompt")
        # Step 2: Use VLM to reasoning the editing area, get two points: Left Top and Right Bottom, and get mask-prompt. Then the image part in box is as image' and will be add a mask
        content = []
        image_resized = Image.open(image_resized_path)  
        buffer = io.BytesIO()  
        image_resized.save(buffer, format="PNG")  
        result_base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        content.append({  
            "type": "image_url",  
            "image_url": {  
                "url": f"data:image/png;base64,{result_base64_image}"  # Correct MIME type for PNG  
            }  
        })
        system_prompt_reason_edit = """
        # Task Description
        You will receive a user instruction (e.g., "Replace the animal on the left with a chicken") and will need to reason based on the provided image.

        ## Processing Steps:
        Interpret the user instruction and the reference image to identify the target for modification;
        Analyze the image to locate the specific area of the target object;
        Generate a prompt describing the object to be modified;

        Prompt_Replaced: Only include the object(optional adj. and noun) to be replaced. Do not add any place adverbials, etc. This will be passed to the Mask Generator.
        Prompt_Added: Only include the object to be added. This will be passed to the Inpainting Generator. 

        ## Requirements
        Output Format: Only the following four tags are allowed in your final output:
        <think> ... </think> (Chain of Thought)
        <Prompt_Replaced> ... </Prompt_Replaced> (Prompt_Replaced)
        <Prompt_Added> ... </Prompt_Added> (Prompt_Added)

        ## ATTENTION
        The user will provide specific instructions later.
        The Prompt must include only the object to be passed to the Mask Generator. Do not include any actions, explanations, descriptions of the replacement object, etc.
            - Incorrect example: Replace the left bird with a realistic chicken while preserving lighting and shadows consistent with the environment.
            - Correct example: Bird
        """  
        content.append({
            "type": "text",
            "text": f'The instruction is {instruction}'
        })
        message = [
            {
                "role": "system",
                "content": system_prompt_reason_edit
            },
            {
                "role": "user",
                "content": content
            }
        ]
        response, usage = self.invoke_vision(message)
        logger.info(f"Response: {response}")
        
        import re
        prompt_replaced_match = re.search(r"<Prompt_Replaced>\s*(.*?)\s*</Prompt_Replaced>", response)
        prompt_replaced = prompt_replaced_match.group(1) if prompt_replaced_match else None
        prompt_added_match = re.search(r"<Prompt_Added>\s*(.*?)\s*</Prompt_Added>", response)
        prompt_added = prompt_added_match.group(1) if prompt_added_match else None

        # Double Check
        double_check_system_prompt = f"""
        # Task Objective
        Read and interpret the user's natural language instruction;
        Accurately identify the single object in the image that corresponds exactly to the instruction;
        From scratch, plan a bounding box that tightly and completely contains the target object.

        # Bounding Box Rules
        You must follow these guidelines to determine the box:
        ## 1. Determine Full Boundaries
        Identify the object's extreme positions in the image:
        Top: the highest point of the object
        Bottom: the lowest point of the object
        Left: the leftmost point of the object
        Right: the rightmost point of the object
        ## 2. Check for Full Containment
        The box must fully include all visible parts of the object;
        The edges must not cut through any part of the object;
        Do not include any unrelated or similar-looking objects.

        # Output Format
        Your response must contain only the following tags:
        <think> ... </think> (Chain of Thought)
        <TopLeft>(x1, y1)</TopLeft>
        <BottomRight>(x2, y2)</BottomRight>
        (x1, y1): coordinates of the top-left corner
        (x2, y2): coordinates of the bottom-right corner
        The first value is horizontal (x), the second is vertical (y)
        Use only integers (no decimals)
        Do not include any explanation, reasoning, or extra text — only the coordinates in the specified tags
        - Given resolution of image is {resized_length}*{resized_length}
        # ATTENTION
        - think step by step, think as much as possible
        """
        content = []
        content.append({  
            "type": "image_url",  
            "image_url": {  
                "url": f"data:image/png;base64,{result_base64_image}"  # Correct MIME type for PNG  
            }  
        })
        content.append({
            "type": "text",
            "text": f"The instruction is {instruction}."
        })
        message = [
            {
                "role": "system",
                "content": double_check_system_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ]
        response, usage = self.invoke_vision_reasoning(message)
        top_left_match = re.search(r"<TopLeft>\(\s*(\d+)\s*,\s*(\d+)\s*\)</TopLeft>", response)
        top_left = [int(top_left_match.group(1)), int(top_left_match.group(2))] if top_left_match else None

        bottom_right_match = re.search(r"<BottomRight>\(\s*(\d+)\s*,\s*(\d+)\s*\)</BottomRight>", response)
        bottom_right = [int(bottom_right_match.group(1)), int(bottom_right_match.group(2))] if bottom_right_match else None
        
        
        # Optimize the prompt_added using LLM
        message = [
            {
                "role": "system",
                "content": """ 
                    Add a detailed description for the prompt_added content. Focus exclusively on the appearance of the replaced object,
                    including its shape, texture, color, materials, surface details, distinctive patterns, and any unique physical features.
                    Do not include any information about actions, style, lighting, environment, or background context. 
                    The description should be concise, objective, and well-structured. Remove redundant adjectives, emotional descriptions, metaphors, examples, and subjective judgments.
                    Retain only core facts.
                    ATTENTION: Don't use pronouns, always use the specific content you want to draw.
                    The output should only include the detailed description of the replaced object, without any other information, and which is wrapped by <Prompt_Added> ... </Prompt_Added>.
                """
            },
            {
                "role": "user",
                "content": f"The prompt_added is {prompt_added}."
            }
        ]
        response, usage = self.invoke_vision(message)
        logger.info(f"Detailed description of the replaced object: {response}")
        # parse the response e.g of response: <Prompt_Added> ... </Prompt_Added>
        prompt_added_match = re.search(r"<Prompt_Added>\s*(.*?)\s*</Prompt_Added>", response)
        prompt_added_ = prompt_added_match.group(1) if prompt_added_match else response
        prompt_added = f'{prompt_added} : {prompt_added_}'
        logger.info(f"Optimized prompt_added: {prompt_added}")


        # Enlarge (absolute resolution)
        top_left[0] = max(0, int(top_left[0] - 0/512*resized_length))
        top_left[1] = max(0, int(top_left[1] - 0/512*resized_length))
        bottom_right[0] = min(resized_length, int(bottom_right[0] + 0/512*resized_length))
        bottom_right[1] = min(resized_length, int(bottom_right[1] + 0/512*resized_length))

        # cut the box area from the image as a new image named image_box, and dont change the original image (top_left and others are absolute position from (0 to 1024))
        top_left_pixel = (int(top_left[0]), int(top_left[1]))
        bottom_right_pixel = (int(bottom_right[0]), int(bottom_right[1]))
        image_box = image_resized.crop((top_left_pixel[0], top_left_pixel[1], bottom_right_pixel[0], bottom_right_pixel[1]))
        image_box.save(image_resized_path.replace('_resized.png', '_box.png'))
        image_box_path = image_resized_path.replace('_resized.png', '_box.png')
        logger.info(f"Box image path: {image_box_path}")
        

        logger.info(f"Start to Mask the image' according to the mask-prompt and convert into a mask of image")
        # Step 3: Mask the image' according to the mask-prompt and convert into a mask of image
        workflow_for_mask_box = self.load_workflow_file('059_api.json')
        workflow_str = json.dumps(workflow_for_mask_box)
        workflow_str = workflow_str.replace('%%IMAGE1%%', image_box_path)
        workflow_str = workflow_str.replace('%%PROMPT1%%', prompt_replaced)
        workflow_for_mask_box = json.loads(workflow_str) 

        client_id = str(uuid.uuid4())  
        output_files = []  # List to hold all output files (images, videos)  
        try:  
            ws_url = f"ws://{COMFYUI_SERVER['host']}:{COMFYUI_SERVER['port']}/ws?clientId={client_id}"  
            ws = websocket.WebSocket()  
            ws.connect(ws_url)  
            prompt_id = self.queue_prompt(workflow_for_mask_box, client_id)["prompt_id"]  
            logger.info("Start to wait for completion".center(80, '-'))
            while True:  
                out = ws.recv()  
                if isinstance(out, str):  
                    message = json.loads(out)  
                    if message["type"] == "executing":  
                        data = message["data"]  
                        if data["node"] is None and data["prompt_id"] == prompt_id:  
                            break  
            logger.info("End wait for completion".center(80, '-'))
            history = self.get_history(prompt_id)[prompt_id]  
            for node_id, node_output in history["outputs"].items():    
                if "images" in node_output:  
                    for image in node_output["images"]:  
                        image_path = self.save_and_get_image(  
                            image["filename"], image["subfolder"], image["type"]  
                        )  
                        if image_path:  
                            output_files.append({"images": image_path})  
        except Exception as e:  
            logging.error(f"Error in generate_using_workflow: {str(e)}")  
            return {"error": str(e)}  
        finally:  
            if "ws" in locals():  
                ws.close()
        
        # Get mask from output_files
        box_mask_path = output_files[0]["images"] # String
        logger.info(f"Box mask path: {box_mask_path}")
        # Merge the mask into a total black mask which has the same size as the original image
        box_mask = Image.open(box_mask_path).convert("RGB")

        # Convert top_left and bottom_right in resized image to top_left and bottom_right in original image
        top_left_pixel = (
            int(top_left[0] * ori_image_width / resized_length),
            int(top_left[1] * ori_image_height / resized_length)
        )
        bottom_right_pixel = (
            int(bottom_right[0] * ori_image_width / resized_length),
            int(bottom_right[1] * ori_image_height / resized_length)
        )
        box_width = bottom_right_pixel[0] - top_left_pixel[0]
        box_height = bottom_right_pixel[1] - top_left_pixel[1]
        if box_mask.size != (box_width, box_height):
            box_mask = box_mask.resize((box_width, box_height))

        full_mask = Image.new("RGB", (ori_image_width, ori_image_height), color=(0, 0, 0))
        full_mask.paste(box_mask, top_left_pixel)
        full_mask.save(image_resized_path.replace('_resized.png', '_maskW.png'))
        full_mask_path = image_resized_path.replace('_resized.png', '_maskW.png')
        logger.info(f"Full mask path(before keep_largest_connected_component_color): {full_mask_path}")

        # keep_largest_connected_component_color
        mask_color = cv2.imread(full_mask_path)
        cleaned_mask = self.keep_largest_connected_component_color(mask_color)
        cv2.imwrite(full_mask_path, cleaned_mask)
        logger.info(f"Full mask path(after keep_largest_connected_component_color): {full_mask_path}")


        # Step 4: Using Inpainting workflow to generate the new image
        workflow_for_inpaint = self.load_workflow_file(f'{workflow_name}')
        workflow_str = json.dumps(workflow_for_inpaint)
        workflow_str = workflow_str.replace('%%IMAGE1%%', ori_image_path)
        workflow_str = workflow_str.replace('%%PROMPT1%%', json.dumps(prompt_added)[1:-1])
        workflow_str = workflow_str.replace('%%MASK1%%', full_mask_path)
        workflow_for_inpaint = json.loads(workflow_str) 

        # Set random seed
        import secrets
        seed_temp = secrets.randbelow(10000000001)
        if f'{workflow_name}' == '053_api.json':
            workflow_for_inpaint["10"]["inputs"]["seed"] = seed_temp
            workflow_for_inpaint["45"]["inputs"]["seed"] = seed_temp
        elif f'{workflow_name}' == '051_api.json':
            workflow_for_inpaint["83"]["inputs"]["seed"] = seed_temp
        elif f'{workflow_name}' == '052_api.json':
            workflow_for_inpaint["14"]["inputs"]["seed"] = seed_temp
            workflow_for_inpaint["34"]["inputs"]["seed"] = seed_temp


        client_id = str(uuid.uuid4())  
        output_files = []   
        try:  
            ws_url = f"{self.ws_base_url}/ws?clientId={client_id}"  
            ws = websocket.WebSocket()  
            ws.connect(ws_url)  
            prompt_id = self.queue_prompt(workflow_for_inpaint, client_id)["prompt_id"]  
            logger.info("Start to wait for completion".center(80, '-'))
            while True:  
                out = ws.recv()  
                if isinstance(out, str):  
                    message = json.loads(out)  
                    if message["type"] == "executing":  
                        data = message["data"]  
                        if data["node"] is None and data["prompt_id"] == prompt_id:  
                            break  
            logger.info("End wait for completion".center(80, '-'))
            history = self.get_history(prompt_id)[prompt_id]  
            for node_id, node_output in history["outputs"].items():    
                if "images" in node_output:  
                    for image in node_output["images"]:  
                        image_path = self.save_and_get_image(  
                            image["filename"], image["subfolder"], image["type"]  
                        )  
                        if image_path:  
                            output_files.append({"images": image_path})  

            logger.info(f"Final outputs: {output_files}")
            return {"outputs": output_files}  # Return all outputs together  
        except Exception as e:  
            logging.error(f"Error in generate_using_workflow: {str(e)}")  
            return {"error": str(e)}  
        finally:  
            if "ws" in locals():  
                ws.close()


    def generate_using_workflow_reason_removal(self,   
                                workflow_name: str,  # Make workflow_name a required parameter
                                requirements_message: Optional[str] = None,
                                prompt: Optional[Dict[str, str]] = None,   
                                input_images: Optional[Dict[str, str]] = None,   
                                input_videos: Optional[Dict[str, str]] = None,   
                                other_parameters: Optional[Dict[str, str]] = None) -> Dict:  
        # Step 1: Image Resize
        # Step 2: Use VLM to reasoning the editing area, get two points: Left Top and Right Bottom, and get mask-prompt. Then the image part in box is as image' and will be add a mask
        # Step 3: Mask the image' according to the mask-prompt and convert into a mask of image
        # Step 4: Using Reason Removal workflow to generate the new image

        # prepare
        instruction = None
        for key, value in prompt.items():
            if key == '%%PROMPT1%%':
                instruction = value

        # If min of width and height of the image is less than 768, then upscale the image by 2x
        for key, value in input_images.items():
            image = Image.open(value)
            if min(image.size) <= 768:
                upscaled_image_path = self.upscale_image(value) # temporary save the upscaled image
                image = Image.open(upscaled_image_path)
                image.save(value.replace('.png', '_upscaled.png')) # save the upscaled image
                input_images[key] = value.replace('.png', '_upscaled.png')
                logger.info(f"Upscaled image has been saved to {input_images[key]}")


        logger.info(f"Start to resize the image")
        # Step 1: Image Resize
        resized_length = 512
        ori_image_path = None
        for key, value in input_images.items():
            image = Image.open(value)
            ori_image = Image.open(value)
            ori_image_path = value
            image = image.resize((resized_length, resized_length))
            # save as another image(dont change the original image), save the new image path as image_resized
            image.save(value.replace('.png', '_resized.png'))
            image_resized_path = value.replace('.png', '_resized.png')

        ori_image_width = ori_image.size[0]
        ori_image_height = ori_image.size[1]
        logger.info(f"Resized image path: {image_resized_path}")

        logger.info(f"Start to use VLM to reasoning the editing area, get two points: Left Top and Right Bottom, and get mask-prompt")
        # Step 2: Use VLM to reasoning the editing area, get two points: Left Top and Right Bottom, and get mask-prompt. Then the image part in box is as image' and will be add a mask
        content = []
        image_resized = Image.open(image_resized_path)  
        buffer = io.BytesIO()  
        image_resized.save(buffer, format="PNG")  
        result_base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        content.append({  
            "type": "image_url",  
            "image_url": {  
                "url": f"data:image/png;base64,{result_base64_image}"  # Correct MIME type for PNG  
            }  
        })
        system_prompt_reason_removal = """
        # Task Description
        You will receive a user instruction (e.g., "Remove the most expensive object in the image") and will need to reason based on the provided image.

        ## Processing Steps:
        Interpret the user instruction and the reference image to identify the target for removal;
        Analyze the image to locate the specific area of the target object;
        Generate a prompt describing the object to be modified;
        Prompt_Removal: Only include the object(optional adj. and noun) to be removed. Do not add any place adverbials, etc. This will be passed to the Mask Generator.

        ## Requirements
        Output Format: Only the following two tags are allowed in your final output:
        <think> ... </think> (Chain of Thought)
        <Prompt_Removal> ... </Prompt_Removal> (Prompt_Removal)

        ## ATTENTION
        The user will provide specific instructions.
        The Prompt must include only the object to be passed to the Mask Generator. Do not include any actions, explanations, descriptions of the removal object, etc.
            - Incorrect example: Remove the left bird with a realistic chicken while preserving lighting and shadows consistent with the environment.
            - Correct example: Bird
        - think step by step
        """  
        content.append({
            "type": "text",
            "text": f'The instruction is {instruction}'
        })
        message = [
            {
                "role": "system",
                "content": system_prompt_reason_removal
            },
            {
                "role": "user",
                "content": content
            }
        ]
        response, usage = self.invoke_vision(message)
        logger.info(f"Response: {response}")
        
        # parse the response e.g of response:
        # <think> ... </think>
        # <Prompt_Removal> ... </Prompt_Removal>
        # <TopLeft>(x1, y1)</TopLeft>
        # <BottomRight>(x2, y2)</BottomRight>
        import re
        logger.info(f"Response: {response}")

        prompt_removal_match = re.search(r"<Prompt_Removal>\s*(.*?)\s*</Prompt_Removal>", response)
        prompt_removal = prompt_removal_match.group(1) if prompt_removal_match else None

        # Get top_left and bottom_right
        double_check_system_prompt = f"""
        # Task Objective
        Read and interpret the user's natural language instruction;
        Accurately identify the single object in the image that corresponds exactly to the instruction;
        From scratch, plan a bounding box that tightly and completely contains the target object.

        # Bounding Box Rules
        You must follow these guidelines to determine the box:
        ## 1. Determine Full Boundaries
        Identify the object's extreme positions in the image:
        Top: the highest point of the object
        Bottom: the lowest point of the object
        Left: the leftmost point of the object
        Right: the rightmost point of the object
        ## 2. Check for Full Containment
        The box must fully include all visible parts of the object;
        The edges must not cut through any part of the object;
        Do not include any unrelated or similar-looking objects.

        # Output Format
        Your response must contain only the following tags:
        <think> ... </think> (Chain of Thought)
        <TopLeft>(x1, y1)</TopLeft>
        <BottomRight>(x2, y2)</BottomRight>
        (x1, y1): coordinates of the top-left corner
        (x2, y2): coordinates of the bottom-right corner
        The first value is horizontal (x), the second is vertical (y)
        Use only integers (no decimals)
        Do not include any explanation, reasoning, or extra text — only the coordinates in the specified tags
        - Given resolution of image is {resized_length}*{resized_length}
        # ATTENTION
        - think step by step, think as much as possible
        """
        content = []
        content.append({  
            "type": "image_url",  
            "image_url": {  
                "url": f"data:image/png;base64,{result_base64_image}"  # Correct MIME type for PNG  
            }  
        })
        content.append({
            "type": "text",
            "text": f"The instruction is {instruction}."
        })
        message = [
            {
                "role": "system",
                "content": double_check_system_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ]
        response, usage = self.invoke_vision_reasoning(message)
        logger.info(f"Response: {response}")
        top_left_match = re.search(r"<TopLeft>\(\s*(\d+)\s*,\s*(\d+)\s*\)</TopLeft>", response)
        top_left = [int(top_left_match.group(1)), int(top_left_match.group(2))] if top_left_match else None

        bottom_right_match = re.search(r"<BottomRight>\(\s*(\d+)\s*,\s*(\d+)\s*\)</BottomRight>", response)
        bottom_right = [int(bottom_right_match.group(1)), int(bottom_right_match.group(2))] if bottom_right_match else None
        logger.info(f"After double check: Top Left: {top_left}, Bottom Right: {bottom_right}")

        # Enlarge (absolute resolution)
        top_left[0] = max(0, int(top_left[0] - 30/512*resized_length))
        top_left[1] = max(0, int(top_left[1] - 30/512*resized_length))
        bottom_right[0] = min(resized_length, int(bottom_right[0] + 30/512*resized_length))
        bottom_right[1] = min(resized_length, int(bottom_right[1] + 30/512*resized_length))

        # cut the box area from the image as a new image named image_box, and dont change the original image (top_left and others are absolute position from (0 to 1024))
        top_left_pixel = (int(top_left[0]), int(top_left[1]))
        bottom_right_pixel = (int(bottom_right[0]), int(bottom_right[1]))
        image_box = image_resized.crop((top_left_pixel[0], top_left_pixel[1], bottom_right_pixel[0], bottom_right_pixel[1]))
        image_box.save(image_resized_path.replace('_resized.png', '_box.png'))
        image_box_path = image_resized_path.replace('_resized.png', '_box.png')
        logger.info(f"Box image path: {image_box_path}")


        logger.info(f"Start to Mask the image' according to the mask-prompt and convert into a mask of image")
        # Step 3: Mask the image' according to the mask-prompt and convert into a mask of image
        workflow_for_mask_box = self.load_workflow_file('059_api.json')
        workflow_str = json.dumps(workflow_for_mask_box)
        workflow_str = workflow_str.replace('%%IMAGE1%%', image_box_path)
        workflow_str = workflow_str.replace('%%PROMPT1%%', prompt_removal)
        workflow_for_mask_box = json.loads(workflow_str) 

        client_id = str(uuid.uuid4())  
        output_files = []  # List to hold all output files (images, videos)  
        try:  
            ws_url = f"{self.ws_base_url}/ws?clientId={client_id}"  
            ws = websocket.WebSocket()  
            ws.connect(ws_url)  
            prompt_id = self.queue_prompt(workflow_for_mask_box, client_id)["prompt_id"]  
            logger.info("Start to wait for completion".center(80, '-'))
            while True:  
                out = ws.recv()  
                if isinstance(out, str):  
                    message = json.loads(out)  
                    if message["type"] == "executing":  
                        data = message["data"]  
                        if data["node"] is None and data["prompt_id"] == prompt_id:  
                            break  
            logger.info("End wait for completion".center(80, '-'))
            history = self.get_history(prompt_id)[prompt_id]  
            for node_id, node_output in history["outputs"].items():    
                if "images" in node_output:  
                    for image in node_output["images"]:  
                        image_path = self.save_and_get_image(  
                            image["filename"], image["subfolder"], image["type"]  
                        )  
                        if image_path:  
                            output_files.append({"images": image_path})  
        except Exception as e:  
            logging.error(f"Error in generate_using_workflow: {str(e)}")  
            return {"error": str(e)}  
        finally:  
            if "ws" in locals():  
                ws.close()
        
        # Get mask from output_files
        box_mask_path = output_files[0]["images"] # String
        logger.info(f"Box mask path: {box_mask_path}")
        # Merge the mask into a total black mask which has the same size as the original image
        box_mask = Image.open(box_mask_path).convert("RGB")

        # Convert top_left and bottom_right in resized image to top_left and bottom_right in original image
        top_left_pixel = (
            int(top_left[0] * ori_image_width / resized_length),
            int(top_left[1] * ori_image_height / resized_length)
        )
        bottom_right_pixel = (
            int(bottom_right[0] * ori_image_width / resized_length),
            int(bottom_right[1] * ori_image_height / resized_length)
        )
        box_width = bottom_right_pixel[0] - top_left_pixel[0]
        box_height = bottom_right_pixel[1] - top_left_pixel[1]
        if box_mask.size != (box_width, box_height):
            box_mask = box_mask.resize((box_width, box_height))

        full_mask = Image.new("RGB", (ori_image_width, ori_image_height), color=(0, 0, 0))
        full_mask.paste(box_mask, top_left_pixel)
        full_mask.save(image_resized_path.replace('_resized.png', '_maskW.png'))
        full_mask_path = image_resized_path.replace('_resized.png', '_maskW.png')
        logger.info(f"Full mask path(before keep_largest_connected_component_color): {full_mask_path}")

        # keep_largest_connected_component_color
        mask_color = cv2.imread(full_mask_path)
        cleaned_mask = self.keep_largest_connected_component_color(mask_color)
        cv2.imwrite(full_mask_path, cleaned_mask)
        logger.info(f"Full mask path(after keep_largest_connected_component_color): {full_mask_path}")


        # Step 4: Using Removal workflow to generate the new image
        workflow_for_removal = self.load_workflow_file('056_api.json')
        workflow_str = json.dumps(workflow_for_removal)
        workflow_str = workflow_str.replace('%%IMAGE1%%', ori_image_path)
        workflow_str = workflow_str.replace('%%PROMPT1%%', prompt_removal) # As negative prompt
        workflow_str = workflow_str.replace('%%MASK1%%', full_mask_path)
        workflow_for_removal = json.loads(workflow_str) 


        client_id = str(uuid.uuid4())  
        output_files = []  # List to hold all output files (images, videos)  
        try:  
            ws_url = f"{self.ws_base_url}/ws?clientId={client_id}"  
            ws = websocket.WebSocket()  
            ws.connect(ws_url)  
            prompt_id = self.queue_prompt(workflow_for_removal, client_id)["prompt_id"]  
            logger.info("Start to wait for completion".center(80, '-'))
            while True:  
                out = ws.recv()  
                if isinstance(out, str):  
                    message = json.loads(out)  
                    if message["type"] == "executing":  
                        data = message["data"]  
                        if data["node"] is None and data["prompt_id"] == prompt_id:  
                            break  
            logger.info("End wait for completion".center(80, '-'))
            history = self.get_history(prompt_id)[prompt_id]  
            for node_id, node_output in history["outputs"].items():    
                if "images" in node_output:  
                    for image in node_output["images"]:  
                        image_path = self.save_and_get_image(  
                            image["filename"], image["subfolder"], image["type"]  
                        )  
                        if image_path:  
                            output_files.append({"images": image_path})  

            logger.info(f"Final outputs: {output_files}")
            return {"outputs": output_files}  # Return all outputs together  
        except Exception as e:  
            logging.error(f"Error in generate_using_workflow: {str(e)}")  
            return {"error": str(e)}  
        finally:  
            if "ws" in locals():  
                ws.close()


    def generate_using_workflow_reason_addition(self,   
                                workflow_name: str,  # Make workflow_name a required parameter
                                requirements_message: Optional[str] = None,
                                prompt: Optional[Dict[str, str]] = None,   
                                input_images: Optional[Dict[str, str]] = None,   
                                input_videos: Optional[Dict[str, str]] = None,   
                                other_parameters: Optional[Dict[str, str]] = None) -> Dict:  
        # Step 1: Image Resize 
        # Step 2: Use VLM to reasoning the area which will be added object, get two points: Left Top and Right Bottom
        # Step 3: Get mask (The area is white, others are black)
        # Step 4: Using Reason Removal workflow to generate the new image
        # prepare
        instruction = None
        for key, value in prompt.items():
            if key == '%%PROMPT1%%':
                instruction = value

        # If min of width and height of the image is less than 768, then upscale the image by 2x
        for key, value in input_images.items():
            image = Image.open(value)
            if min(image.size) <= 768:
                upscaled_image_path = self.upscale_image(value) # temporary save the upscaled image
                image = Image.open(upscaled_image_path)
                image.save(value.replace('.png', '_upscaled.png')) # save the upscaled image
                input_images[key] = value.replace('.png', '_upscaled.png')
                logger.info(f"Upscaled image has been saved to {input_images[key]}")


        logger.info(f"Start to resize the image to 512 * 512")
        # Step 1: Image Resize to 512 * 512
        resized_length = 512
        ori_image_path = None
        for key, value in input_images.items():
            image = Image.open(value)
            ori_image = Image.open(value)
            ori_image_path = value
            image = image.resize((resized_length, resized_length))
            # save as another image(dont change the original image), save the new image path as image_resized
            image.save(value.replace('.png', '_resized.png'))
            image_resized_path = value.replace('.png', '_resized.png')

        ori_image_width = ori_image.size[0]
        ori_image_height = ori_image.size[1]
        logger.info(f"Resized image path: {image_resized_path}")

        logger.info(f"Start to use VLM to reasoning the area which will be added object, get two points: Left Top and Right Bottom")
        # Step 2: Use VLM to reasoning the area which will be added object, get two points: Left Top and Right Bottom
        content = []
        image_resized = Image.open(image_resized_path)  
        buffer = io.BytesIO()  
        image_resized.save(buffer, format="PNG")  
        result_base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        content.append({  
            "type": "image_url",  
            "image_url": {  
                "url": f"data:image/png;base64,{result_base64_image}"  # Correct MIME type for PNG  
            }  
        })

        system_prompt_reason_addition = f"""
        Task Description:
        You will receive a user instruction along with a reference image. The instruction will specify adding an object to a particular region in the image (e.g., "Add a vase to the left of the table"). Your task is to:
        Interpret the instruction and analyze the image to identify a suitable and logical region for placing the specified object.
        Ensure the selected region is appropriate for the object—there should be enough space, and it should not interfere with or cover other important elements in the image.
        Generate a prompt describing only the object to be added (e.g., "vase with red flowers")—this will be used for image inpainting.
        Determine the bounding box of the identified region by providing its top-left and bottom-right coordinates.
        Recognition Region Guidelines:
        Use spatial cues in the instruction (e.g., "on the left", "next to", "above") to determine the intended placement area.
        Make sure the bounding box fully covers the space where the new object will be added and is appropriate in size.
        Avoid overlapping the bounding box with existing key elements in the image.

        Output Format:
        Your response must contain only the following tags:
        <think> ... </think> (Chain of Thought)
        <Prompt_Added> ... </Prompt_Added>
        <TopLeft>(x1, y1)</TopLeft>
        <BottomRight>(x2, y2)</BottomRight>
        (x1, y1): coordinates of the top-left corner
        (x2, y2): coordinates of the bottom-right corner
        The first value is horizontal (x), the second is vertical (y)

        Use only integers (no decimals)
        Do not include any explanation, reasoning, or extra text — only the coordinates in the specified tags

        ATTENTION:
        - think step by step, think as much as possible
        - The placement should be logical, not in the air, and the size should be appropriate. (The cats and dogs placed are smaller than the elephants)
        - Given resolution of image is {resized_length}*{resized_length}
        """  
        content.append({
            "type": "text",
            "text": f'The instruction is {instruction}'
        })
        message = [
            {
                "role": "system",
                "content": system_prompt_reason_addition
            },
            {
                "role": "user",
                "content": content
            }
        ]
        response, usage = self.invoke_vision(message)
        logger.info(f"Response: {response}")
        
        # parse the response e.g of response:
        # <think> ... </think>
        # <Prompt> ... </Prompt>
        # <TopLeft>(x1, y1)</TopLeft>
        # <BottomRight>(x2, y2)</BottomRight>
        import re
        prompt_added_match = re.search(r"<Prompt_Added>\s*(.*?)\s*</Prompt_Added>", response)
        prompt_added = prompt_added_match.group(1) if prompt_added_match else None

        top_left_match = re.search(r"<TopLeft>\(\s*(\d+)\s*,\s*(\d+)\s*\)</TopLeft>", response)
        top_left = [int(top_left_match.group(1)), int(top_left_match.group(2))] if top_left_match else None

        bottom_right_match = re.search(r"<BottomRight>\(\s*(\d+)\s*,\s*(\d+)\s*\)</BottomRight>", response)
        bottom_right = [int(bottom_right_match.group(1)), int(bottom_right_match.group(2))] if bottom_right_match else None

        # No double check
        logger.info(f"Top Left: {top_left}, Bottom Right: {bottom_right}")
        logger.info(f"Prompt_added: {prompt_added}")

        # Optimize the prompt_added using LLM
        message = [
            {
                "role": "system",
                "content": """ 
                    Add a detailed description for the prompt_added content. Focus exclusively on the appearance of the replaced object,
                    including its shape, texture, color, materials, surface details, distinctive patterns, and any unique physical features.
                    Do not include any information about actions, style, lighting, environment, or background context. 
                    The description should be concise, objective, and well-structured. Remove redundant adjectives, emotional descriptions, metaphors, examples, and subjective judgments.
                    Retain only core facts.
                    ATTENTION: Don't use pronouns, always use the specific content you want to draw.
                    The output should only include the detailed description of the replaced object, without any other information, and which is wrapped by <Prompt_Added> ... </Prompt_Added>.
                """
            },
            {
                "role": "user",
                "content": f"The prompt_added is {prompt_added}."
            }
        ]
        response, usage = self.invoke_vision(message)
        logger.info(f"Detailed description of the replaced object: {response}")
        # parse the response e.g of response: <Prompt_Added> ... </Prompt_Added>
        prompt_added_match = re.search(r"<Prompt_Added>\s*(.*?)\s*</Prompt_Added>", response)
        prompt_added_ = prompt_added_match.group(1) if prompt_added_match else response
        prompt_added = f'{prompt_added} : {prompt_added_}'
        logger.info(f"Optimized prompt_added: {prompt_added}")


        top_left_pixel = (
            int(top_left[0] * ori_image_width / resized_length),
            int(top_left[1] * ori_image_height / resized_length)
        )
        bottom_right_pixel = (
            int(bottom_right[0] * ori_image_width / resized_length),
            int(bottom_right[1] * ori_image_height / resized_length)
        )

        # Create mask image (black background)
        mask_image = Image.new("RGB", (ori_image_width, ori_image_height), color=(0, 0, 0))

        # Create white rectangle and paste it
        white_rect = Image.new(
            "RGB",
            (bottom_right_pixel[0] - top_left_pixel[0], bottom_right_pixel[1] - top_left_pixel[1]),
            color=(255, 255, 255)
        )
        mask_image.paste(white_rect, top_left_pixel)
        mask_image.save(image_resized_path.replace('_resized.png', '_maskW.png'))
        full_mask_path = image_resized_path.replace('_resized.png', '_maskW.png')
        logger.info(f"Full mask path: {full_mask_path}")


        # Step 4: Using Addition workflow to generate the new image
        workflow_for_addition = self.load_workflow_file(workflow_name) # use inpainting workflow and edit it, 054_api.json or 055_api.json
            
        workflow_str = json.dumps(workflow_for_addition)
        workflow_str = workflow_str.replace('%%IMAGE1%%', ori_image_path)
        workflow_str = workflow_str.replace('%%PROMPT1%%', json.dumps(prompt_added)[1:-1])
        workflow_str = workflow_str.replace('%%MASK1%%', full_mask_path)
        workflow_for_addition = json.loads(workflow_str) 

        client_id = str(uuid.uuid4())  
        output_files = []  
        try:  
            ws_url = f"{self.ws_base_url}/ws?clientId={client_id}"  
            ws = websocket.WebSocket()  
            ws.connect(ws_url)  
            prompt_id = self.queue_prompt(workflow_for_addition, client_id)["prompt_id"]  
            logger.info("Start to wait for completion".center(80, '-'))
            while True:  
                out = ws.recv()  
                if isinstance(out, str):  
                    message = json.loads(out)  
                    if message["type"] == "executing":  
                        data = message["data"]  
                        if data["node"] is None and data["prompt_id"] == prompt_id:  
                            break  
            logger.info("End wait for completion".center(80, '-'))
            history = self.get_history(prompt_id)[prompt_id]  
            for node_id, node_output in history["outputs"].items():    
                if "images" in node_output:  
                    for image in node_output["images"]:  
                        image_path = self.save_and_get_image(  
                            image["filename"], image["subfolder"], image["type"]  
                        )  
                        if image_path:  
                            output_files.append({"images": image_path})  

            logger.info(f"Final outputs: {output_files}")
            return {"outputs": output_files}  # Return all outputs together  
        except Exception as e:  
            logging.error(f"Error in generate_using_workflow: {str(e)}")  
            return {"error": str(e)}  
        finally:  
            if "ws" in locals():  
                ws.close()        


    def generate_using_workflow_reason_t2i_generation(self,   
                                workflow_name: str,  # Make workflow_name a required parameter
                                requirements_message: Optional[str] = None,
                                prompt: Optional[Dict[str, str]] = None,   
                                input_images: Optional[Dict[str, str]] = None,   
                                input_videos: Optional[Dict[str, str]] = None,   
                                other_parameters: Optional[Dict[str, str]] = None) -> Dict:  
        system_reasoning_t2i_generation = f"""
        You are an expert in visual reasoning and prompt transformation. You are an encyclopedic scholar, and you know very well what prompts the generative model needs to improve its generation quality. Your task is to convert general knowledge, scientific, or spatio-temporal prompts into **concrete and explicit image generation descriptions** that an image generation model can understand.
        Please strictly follow these rules, think step by step:

        ## Reasoning
        Carefully read the given prompt, and identify the main concepts, nouns, verbs, place names, personal names, time expressions, and other relevant keywords. List these keywords to provide a foundation for the next step of knowledge retrieval.
        Based on the extracted keywords, recall any related common knowledge, scientific principles, laws, or widely accepted facts you are familiar with.
        If the prompt includes specific time points, events, or historical periods, organize these chronologically in your response.
        If the prompt involves locations (such as countries, cities, geographic features, etc.), connect them with relevant geographic, historical, cultural, or climatic information.
        After extracting and analyzing the information, summarize them.
        Use a structured format (e.g., bullet points or numbered lists) to present this knowledge structure, so it can be referenced clearly during subsequent reasoning and answering.
        You should reason about the behavior, appearance, and growth patterns of animals and plants across different times and seasons. These should align with their natural habits and common sense. For example: most plants do not bloom in winter.
        Ignore any irrelevant details in the prompt to avoid straying off-topic in your response.
        Based on all the knowledge you have above, infer what needs to be generated in the instructions. The final inference result should be able to be summarized with a specific phrase vocabulary, rather than a vague reference (such as an animal, a craft)

        ## Scientific Consistency Check:
        Before generating the image, ensure that the scenario adheres to fundamental scientific principles across physics, chemistry, and biology. Clearly infer the expected observable outcomes—such as material changes,
        structural transformations, or characteristic biological responses—and accurately reflect the relevant temporal state (e.g., “freshly cut,” “decayed,” or “after being touched”). For scenarios where no effect should occur,
        depict the failure or lack of change explicitly rather than assuming success. Plants and animals should exhibit natural, context-appropriate behaviors rather than static or symbolic representations. All content must be
        grounded in real-world logic, avoiding fictional or speculative elements unless explicitly requested by the user. When animals or plants are involved, their behavior, appearance, and state must align with seasonal cycles, environmental conditions, and known responses to stimuli or threats. 
        When it comes to physical chemistry, Always verify the material properties and physical thresholds (e.g., boiling points, conductivity) involved in the scenario, and ensure the visual outcome reflects these real-world constraints accurately.
        For chemical reactions or combustion, ensure the visual outcome reflects real material behavior under correct conditions, not oversimplified assumptions.
        
        Attention : The final result of the reasoning corresponds to the content that exists in reality, not fiction. Note that your goal is not to design fictional content according to instructions, but to infer elements that exist in reality.

        ## Get Prompt for Generation model
        Use common sense, scientific facts, temporal logic, or geographical knowledge to identify what must be visually shown.
        If the user's instruction involves perspective, be sure to describe it according to physical and real-world principles.
        Prompt should conform to spatial relationships and clearly define size relations.
        All content requested by the user should be clearly presented; do not blur the foreground or background due to focus.
        If the user mentions occlusion relationships, they must be strictly followed and transformed into visual generation prompt form.
        Make all visual elements explicit. List out the required objects, their attributes (color, size, material, action), and environmental features (e.g., time of day, season, lighting).
        Clarify spatial or temporal relationships. If the prompt involves spatial layout, time zones, seasons, or historical context, be sure to reflect these explicitly in the visual output.
        Do not use subjective or abstract terms. Focus on objective, observable visual cues. 
        As long as it does not conflict with the user's instruction or previous reasoning, the image prompt should be optimized to achieve high-quality generation results. Specifically, enhancements such as improved lighting and fine details may be added.
        Avoid using empty or pure white backgrounds unless explicitly specified. Backgrounds should include relevant, realistic environmental context consistent with the subject matter (e.g., laboratory settings, natural ecosystems, historical architecture). The background must enhance visual richness and aesthetic appeal, using coherent lighting, textures, and spatial depth to support the primary subject without distraction.
        Attention : By default, the image style must be realistic and photographic, unless the user explicitly requests a specific style.

        ## Very Important Attention:
        - The final prompt should *remove directive/instruction sentences*, such as "Generate an image of...," as well as general summary statements. It should consist solely of the direct, specific, and visually descriptive prompt.
        - The final prompt should not include any negative statements such as "do not include...," as these can lead to misunderstandings. It should directly describe the intended visual scene.

        ## Output in a strict format:
        <think> Chain of thought</think>
        <prompt> The output prompt</prompt>
        """  

        for key, value in prompt.items():
            if key == '%%PROMPT1%%':
                prompt_temp = value
        content = []
        content.append({
            "type": "text",
            "text": f'The prompt is: {prompt_temp}'
        })
        message = [
            {
                "role": "system",
                "content": system_reasoning_t2i_generation
            },
            {
                "role": "user",
                "content": content
            }
        ]
        response, usage = self.invoke_vision_reasoning(message, model_name=TEXT_REASONING_MODEL_NAME, api_key=TEXT_REASONING_API_KEY, base_url=TEXT_REASONING_BASE_URL)
        logger.info(f"Response: {response}")
        

        import re
        thinking = re.search(r"<think>\s*(.*?)\s*</think>", response)
        thinking = thinking.group(1) if thinking else None

        prompt_optimized = re.search(r"<prompt>\s*(.*?)\s*</prompt>", response)
        prompt_optimized = prompt_optimized.group(1) if prompt_optimized else None

        logger.info(f"Thinking: {thinking}")
        logger.info(f"Prompt_optimized: {prompt_optimized}")

        for key, value in prompt.items():
            if key == '%%PROMPT1%%':
                prompt[key] = prompt_optimized if prompt_optimized is not None else prompt[key]
                
        if workflow_name == '057_api.json':
            workflow_name = '031_api.json'
        elif workflow_name == '058_api.json':
            workflow_name = '048_api.json'
        else :
            workflow_name = '031_api.json'
        return self.generate_using_workflow(workflow_name, requirements_message, prompt, input_images, input_videos, other_parameters)


    def generate_using_workflow_prompt_optimization_t2i_generation(self,   
                                workflow_name: str,  # Make workflow_name a required parameter
                                requirements_message: Optional[str] = None,
                                prompt: Optional[Dict[str, str]] = None,   
                                input_images: Optional[Dict[str, str]] = None,   
                                input_videos: Optional[Dict[str, str]] = None,   
                                other_parameters: Optional[Dict[str, str]] = None) -> Dict:  
        
        system_prompt_optimization_t2i_generation = f"""
        You are an expert in optimizing image generation prompts, specifically for task-oriented generation. Your goal is to think step by step and revise or enhance the user’s original prompt to ensure that **each object is clearly and fully generated** and can be **reliably recognized by object detection models**. When rewriting or optimizing the original prompt, strictly follow the guidelines below:
        1. **Structurally express all object information**. Explicitly list each object class individually. Do not merge, generalize, or omit any category. Each class must be clearly and separately mentioned in the prompt.
        
        2. **Add typical color and shape descriptions for each object class**. If the user specifies the color of the object, avoid introducing other colors to avoid confusion. Try to add lots of typical details of the object in the description and emphasize the user's specified color multiple times even if the color is not commonly seen in the object. Use the common and representative appearance features of the object to improve recognizability. Each object should be described in detail. Avoid introducing any ambiguity or conflict with the user's original description. *The descriptions added to the object material, etc. must not conflict with the user-specified colors*
        
        - Attention : Make sure every part of the object in the description is the required color

        3. Add phrases: “placed apart from each other”

        4. **Explicitly emphasize the quantity of each object**. If the user specifies the number of objects, reinforce it with clear expressions like:
        - “3 apples”
        - “2 chairs”

        5. **Default to a simple background**, unless the user explicitly specifies otherwise. You may use:
        - “plain background”
        - “minimal background”

        6. **Specify the style as realistic/photo-like**. Use phrases like:
        - “photorealistic style”
        - “realistic textures”  
        This improves the chances of objects being detected by recognition models. Attention: The specified background color should be highly distinguishable from the color of the content to be generated. 

        You must not change the user’s original intent or add subjective embellishments. Only make objective additions that **enhance detectability and clarity**.

        **Output format:**
        <think>Reasoning process</think>  
        <prompt>Optimized prompt</prompt>

        The user will now provide a prompt.
        """  


        for key, value in prompt.items():
            if key == '%%PROMPT1%%':
                prompt_temp = value
        content = []
        content.append({
            "type": "text",
            "text": f'The prompt is: {prompt_temp}'
        })
        message = [
            {
                "role": "system",
                "content": system_prompt_optimization_t2i_generation
            },
            {
                "role": "user",
                "content": content
            }
        ]
        response, usage = self.invoke_vision(message)
        logger.info(f"Response: {response}")
        

        import re
        thinking = re.search(r"<think>\s*(.*?)\s*</think>", response)
        thinking = thinking.group(1) if thinking else None

        prompt_optimized = re.search(r"<prompt>\s*(.*?)\s*</prompt>", response)
        prompt_optimized = prompt_optimized.group(1) if prompt_optimized else None

        logger.info(f"Thinking: {thinking}")
        logger.info(f"Prompt_optimized: {prompt_optimized}")

        for key, value in prompt.items():
            if key == '%%PROMPT1%%':
                prompt[key] = prompt_optimized if prompt_optimized is not None else prompt[key]
                
        if workflow_name == '049_api.json':
            workflow_name = '031_api.json'
        elif workflow_name == '050_api.json':
            workflow_name = '048_api.json'
        else :
            workflow_name = '031_api.json'
        return self.generate_using_workflow(workflow_name, requirements_message, prompt, input_images, input_videos, other_parameters)


    def generate_using_workflow(self,   
                                workflow_name: str,  # Make workflow_name a required parameter
                                requirements_message: Optional[str] = None,
                                prompt: Optional[Dict[str, str]] = None,   
                                input_images: Optional[Dict[str, str]] = None,   
                                input_videos: Optional[Dict[str, str]] = None,   
                                other_parameters: Optional[Dict[str, str]] = None) -> Dict:  
        """  
        Generate Image or Video using a specified workflow.  

        Args:  
            prompt: An optional dictionary containing text prompts for generation   
                    (e.g., {'%%PROMPT1%%': 'cat'})  
            workflow_name: The name of the workflow to use (required)
            requirements_message: An optional string containing additional requirements for the workflow
            input_images: An optional dictionary of input images paths   
                        (e.g., {'%%IMAGE1%%': '1.png'})  
            input_videos: An optional dictionary of input videos paths   
                        (e.g., {'%%VIDEO1%%': '1.mp4'})  
            other_parameters: An optional dictionary for additional workflow parameters   
                            with string keys and string values   
                            (e.g., {'%%GUIDANCE_SCALE%%': '7.5'})  

        Returns:  
            Dictionary containing generated image and video paths or error message  
        """  

        logger.info("Start to load workflow".center(80, '-'))
        
        workflow = self.load_workflow_file(workflow_name)  
        workflow_meta_info = self.Workflow_meta_info[workflow_name]

        logger.info("End load workflow".center(80, '-'))


        if isinstance(workflow, dict) and "error" in workflow:  
            return workflow  

        client_id = str(uuid.uuid4())  
        output_files = []  

        try:  
            # Connect WebSocket  
            ws_url = f"{self.ws_base_url}/ws?clientId={client_id}"  
            ws = websocket.WebSocket()  
            ws.connect(ws_url)  

            logger.info("Start to replace placeholders".center(80, '-'))

            # Prepare workflow  
            workflow = self.replace_placeholders(workflow, prompt, input_images, input_videos, other_parameters)

            logger.info("End replace placeholders".center(80, '-'))

            if requirements_message is not None and requirements_message != ' ' and requirements_message != '':
               logger.info(f"requirements_message: {requirements_message}")
               logger.info(f"requirements_message: {requirements_message}")
               workflow = self.fit_requirements(workflow, prompt, input_images, input_videos, other_parameters, requirements_message)

            prompt_id = self.queue_prompt(workflow, client_id)["prompt_id"]  


            # Wait for completion  
            logger.info("Start to wait for completion".center(80, '-'))
            while True:  
                out = ws.recv()  
                if isinstance(out, str):  
                    message = json.loads(out)  
                    if message["type"] == "executing":  
                        data = message["data"]  
                        if data["node"] is None and data["prompt_id"] == prompt_id:  
                            break  

            logger.info("End wait for completion".center(80, '-'))
            history = self.get_history(prompt_id)[prompt_id]  

            for node_id, node_output in history["outputs"].items():  

                # Check and process video outputs  
                if "gifs" in node_output:  
                    for video in node_output["gifs"]:  
                        video_path = self.save_and_get_image(  
                            video["filename"], video["subfolder"], video["type"]  
                        )  
                        if video_path:  
                            output_files.append({"videos": video_path})  

                # Check and process image outputs  
                if "images" in node_output:  
                    for image in node_output["images"]:  
                        image_path = self.save_and_get_image(  
                            image["filename"], image["subfolder"], image["type"]  
                        )  
                        if image_path:  
                            output_files.append({"images": image_path})  

            return {"outputs": output_files}  # Return all outputs together  

        except Exception as e:  
            logging.error(f"Error in generate_using_workflow: {str(e)}")  
            return {"error": str(e)}  
        finally:  
            if "ws" in locals():  
                ws.close()


    def replace_placeholders(self, workflow, prompt, input_images, input_videos, other_parameters):
        """Helper method to replace placeholders in the workflow."""
        workflow_str = json.dumps(workflow)
        if prompt is not None:
            for key, value in prompt.items():
                if isinstance(value, str):
                    replacement = json.dumps(value)[1:-1]
                    workflow_str = workflow_str.replace(key, replacement)
                else:
                    workflow_str = workflow_str.replace(key, value)

        if input_images is not None:
            for key, value in input_images.items():
                workflow_str = workflow_str.replace(key, value)

        if input_videos is not None:
            for key, value in input_videos.items():
                workflow_str = workflow_str.replace(key, value)

        if other_parameters is not None:
            for key, value in other_parameters.items():
                if key is not None:
                    workflow_str = workflow_str.replace(key, value)

        #print(f"workflow: {json.loads(workflow_str)}")
        return json.loads(workflow_str)  # Return the modified workflow


class Tools:
    def __init__(self):
        self.comfy_ui = ComfyUITools()

    def generate_using_workflow(self, workflow_name: str, requirements_message: Optional[str] = None, prompt: Optional[Dict[str, str]] = None, input_images: Optional[Dict[str, str]] = None, input_videos: Optional[Dict[str, str]] = None, other_parameters: Optional[Dict[str, str]] = None) -> Dict:
        if workflow_name == '051_api.json' or workflow_name == '052_api.json' or workflow_name == '053_api.json':
            return self.comfy_ui.generate_using_workflow_reason_inpainting(workflow_name, requirements_message, prompt, input_images, input_videos, other_parameters)
        elif workflow_name == '056_api.json':
            return self.comfy_ui.generate_using_workflow_reason_removal(workflow_name, requirements_message, prompt, input_images, input_videos, other_parameters)
        elif workflow_name == '054_api.json' or workflow_name == '055_api.json':
            return self.comfy_ui.generate_using_workflow_reason_addition(workflow_name, requirements_message, prompt, input_images, input_videos, other_parameters)
        elif workflow_name == '058_api.json' or workflow_name == '057_api.json':
            return self.comfy_ui.generate_using_workflow_reason_t2i_generation(workflow_name, requirements_message, prompt, input_images, input_videos, other_parameters)
        elif workflow_name == '049_api.json' or workflow_name == '050_api.json':
            return self.comfy_ui.generate_using_workflow_prompt_optimization_t2i_generation(workflow_name, requirements_message, prompt, input_images, input_videos, other_parameters)
        else:
            return self.comfy_ui.generate_using_workflow(workflow_name, requirements_message, prompt, input_images, input_videos, other_parameters)
