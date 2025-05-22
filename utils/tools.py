import requests
import json
import random
import logging
logger = logging.getLogger(__name__) 

from utils.tools_for_comfyui import Tools
import numpy as np

# Instantiate ComfyUITools
comfy_ui_tools = Tools()

# Predefined function definitions
tools = [
    {
        "name": "Generate_Using_Workflow",
        "description": "Generate Image or Video by workflow",
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_name": {
                    "type": "string",
                    "description": "Name of the workflow to use"
                },
                "prompt": {
                    "type": "dist",
                    "description": "Text prompt for generation. The specific number and dictionary key depends on the selected workflow requirements. Example: {'%%PROMPT1%%': 'cat', '%%PROMPT2%%': 'dog'}"
                },
                "input_images": {
                    "type": "dist",
                    "description": "Path to input images. The specific number and dictionary key depends on the selected workflow requirements. Example: {'%%IMAGE1%%': '1.png', '%%IMAGE2%%': '2.jpeg'}"
                },
                "input_videos": {
                    "type": "dist",
                    "description": "Path to input videos. The specific number and dictionary key depends on the selected workflow requirements. Example: {'%%VIDEO1%%': '1.mp4', '%%VIDEO2%%': '2.gif'}"
                }
            },
            "required(ALL PARAMETERS ARE OPTIONAL EXCEPT WORKFLOW_NAME)": [
                "workflow_name",
                "prompt",
                "input_images",
                "input_videos"
            ]
        }
    },
]

# Function mapping collection
functions = {
    "Generate_Using_Workflow": comfy_ui_tools.generate_using_workflow,
}

TOOLS = tools  # Directly reference the previously defined tools list
FUNCTIONS = functions  # Directly reference the previously defined functions dictionary

def extract_json(s):
    stack = 0
    start = s.find('{')
    if start == -1:
        return None

    for i in range(start, len(s)):
        if s[i] == '{':
            stack += 1
        elif s[i] == '}':
            stack -= 1
            if stack == 0:
                return s[start:i + 1]
    return None