import os
import json
import yaml
import re
import hashlib
import logging
logger = logging.getLogger(__name__) 
from typing import Union, Tuple, Any
from openai import OpenAI
from utils.tools import TOOLS, FUNCTIONS


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



def invoke_vision_reasoning(message: any, model_name = None, api_key = None, base_url = None) -> tuple[str, any]:
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
        response = client.chat.completions.create(
            model=VISION_REASONING_MODEL_NAME,
            messages=message
        )
        logger.info(f"Reasoning VisionLM call success")
        answer = response.choices[0].message.content
        usage = response.usage

    except Exception as error:
        answer = f'Error: {error}'
        usage = None

    return answer, usage


def invoke_text_reasoning(message: any, model_name = None, api_key = None, base_url = None) -> tuple[str, any]:
    if base_url and api_key:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    else:
        client = OpenAI(
            base_url=TEXT_REASONING_BASE_URL,
            api_key=TEXT_REASONING_API_KEY
        )

    try:
        response = client.chat.completions.create(
            model=TEXT_REASONING_MODEL_NAME,
            messages=message
        )
        logger.info(f"Reasoning LLM call success")
        answer = response.choices[0].message.content
        usage = response.usage

    except Exception as error:
        answer = f'Error: {error}'
        usage = None

    return answer, usage


def invoke_text(messages, model=COMPLETION_MODEL_NAME):  
    """  
    Single LLM call  
    
    Args:  
        messages (list): Dialog message list  
        model (str): Model name  
    
    Returns:  
        str: LLM's text response  
    """  
    client = OpenAI(  
        base_url=COMPLETION_BASE_URL,  
        api_key=COMPLETION_API_KEY  
    )  
    
    try:  
        response = client.chat.completions.create(
            model=COMPLETION_MODEL_NAME,
            messages=messages,
            functions=TOOLS,
            **COMPLETION_HYPER_PARAMETER
        )
        logger.info(f"LLM call success")
        return response.choices[0].message.content  
    except Exception as e:  
        logger.info(f"LLM call error: {e}")
        return None  


def invoke_vision(messages, model=VISION_MODEL_NAME):
    client = OpenAI(  
        base_url=VISION_BASE_URL,  
        api_key=VISION_API_KEY  
    )  
    
    try:  
        response = client.chat.completions.create(
            model=VISION_MODEL_NAME,
            messages=messages,
            **VISION_HYPER_PARAMETER
        )
        logger.info(f"VisionLM call success")
        return response.choices[0].message.content  
    except Exception as e:  
        logger.info(f"VisionLM call error: {e}")
        return None  


