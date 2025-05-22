import io
import os
import sys
import time
import json
import base64
import argparse
import multiprocessing as mp
from copy import deepcopy

from openai import OpenAI
import cv2
import pandas as pd
from PIL import Image
from bs4 import BeautifulSoup
import yaml

import logging
logger = logging.getLogger(__name__) 

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

t2i_prompt = '''
You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of a text-to-image generation task. You should be tolerant to the quality of the generation result, and focus on the consistency with the instruction.

The task instruction is described as: {instruction}

The given image is the generation result, with an actual resolution of {result_resolution}.

First, analyze whether the generation result meets each key point in the instruction. Enclose your analysis in the <analysis> tag. For example: <analysis>There is a cat in an astronaut suit, which is consistent with the instruction. The wall is white, which is different from the "green wall" in the instruction.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.
'''

t2i_prompt_reasoning = '''
You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of a text-to-image generation task. You should be tolerant to the quality of the generation result, and focus on the consistency with the instruction.

The task instruction is described as: {instruction}

The given image is the generation result, with an actual resolution of {result_resolution}.

First, you need to reason through the user's instruction and truly understand what the user intends to generate. The content must align with general common sense, scientific knowledge, or spatio-temporal logic.
All elements required by the instruction must be clearly and explicitly presented in the image.
If the instruction includes a specific viewpoint, the output should match the indicated perspective and reflect accurate size relationships.
If the instruction mentions occlusion relationships, these should be strictly followed in the image.
If a specific time or season is mentioned, the generated image should respect the typical behavior, appearance, and growth patterns of animals and plants in that season.
The final judgment should be loose, as long as the generated content is reasonable

Enclose your analysis in the <analysis> tag. For example: <analysis>There is a cat in an astronaut suit, which is consistent with the instruction. The wall is white, which is different from the "green wall" in the instruction.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.
'''


i2i_prompt = '''
You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of an image-to-image generation task. You should be tolerant to the quality of the generation result, and focus on the consistency with the instruction.

The task instruction is described as: {instruction}

The first image is the input reference, with an actual resolution of {reference_resolution}. The second image is the generation result, with an actual resolution of {result_resolution}.

First, analyze whether the generation result meets each key point in the instruction based on the input reference. Enclose your analysis in the <analysis> tag. For example: <analysis>The generation result keeps the structure of the input reference, but the car is not removed, which is not consistent with the instruction.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.
'''


t2v_prompt = '''
You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of a text-to-video generation task. You should be tolerant to the quality of the generation result, and focus on the consistency with the instruction.

The task instruction is described as: {instruction}

The given {result_frame_count} images are the frames sampled from the generation result, with an actual resolution of {result_resolution}, duration of {result_duration} seconds and {result_frame_rate} frames per second.

First, analyze whether the generation result meets each key point in the instruction. Enclose your analysis in the <analysis> tag. For example: <analysis>There is a walking robot, which is consistent with the instruction. However, the scene is a street, which is different from the "forest" in the instruction.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.
'''


i2v_prompt = '''
You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of an image-to-video generation task. You should be tolerant to the quality of the generation result, and focus on the consistency with the instruction.

The task instruction is described as: {instruction}

The first image is the input reference, with an actual resolution of {reference_resolution}. The remaining {result_frame_count} images are the frames sampled from the generation result, with an actual resolution of {result_resolution}, duration of {result_duration} seconds and {result_frame_rate} frames per second.

First, analyze whether the generation result meets each key point in the instruction based on the input reference. Enclose your analysis in the <analysis> tag. For example: <analysis>The generation result contains a moving car, which is consistent with the instruction. However, it fails to follow the style of the input reference.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.
'''


v2v_prompt = '''
You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of a video-to-video generation task. You should be tolerant to the quality of the generation result, and focus on the consistency with the instruction.

The task instruction is described as: {instruction}

The first {reference_frame_count} images are the frames sampled from the input reference, with an actual resolution of {reference_resolution}, duration of {reference_duration} seconds and {reference_frame_rate} frames per second. The remaining {result_frame_count} images are the frames sampled from the generation result, with an actual resolution of {result_resolution}, duration of {result_duration} seconds and {result_frame_rate} frames per second.

First, analyze whether the generation result meets each key point in the instruction based on the input reference. Enclose your analysis in the <analysis> tag. For example: <analysis>The generation result improves the resolution of the input reference. However, it fails to convert the input inference into an oil painting style, which is not consistent with the instruction.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.
'''


reason_edit_prompt = '''
You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on the given rules. You will have to give your output in this way (Keep your reasoning concise and short.): ”score” : [...], ”reasoning” : ”...” and don’t output anything else.

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first. The objective is to evaluate how successfully the editing instruction has been executed in the second image.

Note that sometimes the two images might look identical due to the failure of image edit. From a scale 0 to 10: 

A score from 0 to 10 will be given based on the success of the editing. 
- 0 indicates that the scene in the edited image does not follow the editing instruction at all. 
- 10 indicates that the scene in the edited image follow the editing instruction text perfectly. 
- If the object in the instruction is not present in the original image at all, the score will be 0. 

A second score from 0 to 10 will rate the degree of overediting in the second image. 
- 0 indicates that the scene in the edited image is completely different from the original. 
- 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original. 
Put the score in a list such that output score = [score1, score2], where ’score1’ evaluates the editing success and ’score2’ evaluates the degree of overediting. 

Editing instruction: {instruction}. The first image is the source Image, and the second image is the edited Image.
'''



def invoke_vision(message: any) -> tuple[str, any]:
    client = OpenAI(
        base_url=VISION_BASE_URL,
        api_key=VISION_API_KEY
    )

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL_NAME,
            messages=message,
            **VISION_HYPER_PARAMETER
        )
        answer = response.choices[0].message.content
        usage = response.usage

    except Exception as error:
        answer = f'Error: {error}'
        usage = None

    return answer, usage


def encode_image(image: Image.Image, size: tuple[int, int] = (512, 512)) -> str:
    image.thumbnail(size)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_image

def load_image(image_path: str, size_limit: tuple[int, int] = (512, 512)) -> tuple[str, dict]:
    meta_info = {}
    image = Image.open(image_path)
    meta_info['width'], meta_info['height'] = image.size
    base64_image = encode_image(image, size_limit)
    return base64_image, meta_info


def load_video(video_path: str, size_limit: tuple[int, int] = (512, 512), frame_limit: int = 5) -> tuple[list, dict]:
    base64_frames = []
    meta_info = {}
    video = cv2.VideoCapture(video_path)
    meta_info['width'] = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    meta_info['height'] = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meta_info['num_frames'] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    meta_info['frame_rate'] = int(video.get(cv2.CAP_PROP_FPS))
    meta_info['duration'] = meta_info['num_frames'] / meta_info['frame_rate']

    count = 0
    sample_interval = max(6, meta_info['num_frames'] // frame_limit)
    while video.isOpened():
        status, frame = video.read()
        if not status:
            break
        if count % sample_interval == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            base64_frame = encode_image(image, size_limit)
            base64_frames.append(base64_frame)
        count += 1
    video.release()
    return base64_frames, meta_info


def safe_extract_from_soup(soup: BeautifulSoup, tag: str) -> str:
    element = soup.find(tag)
    if element is None:
        return ''
    return element.text.strip()


def parse_evaluation(evaluation: str) -> tuple[str, str]:
    soup = BeautifulSoup(evaluation, 'html.parser')
    analysis = safe_extract_from_soup(soup, 'analysis')
    judgment = safe_extract_from_soup(soup, 'judgment')
    return analysis, judgment


def evaluate_reason_edit(task: dict) -> bool:
    """GPT Score in Reason-Edit"""

    reference_base64_image, reference_meta_info = load_image(f'{task["input_image"]}')
    result_base64_image, result_meta_info = load_image(f'{task["output_image"]}')

    prompt = reason_edit_prompt.format(
        instruction=task['instruction'],
    )

    print(' Evaluator Prompt '.center(80, '-'))
    print(prompt)
    print()

    content = [{
        'type': 'text',
        'text': prompt
    }]
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{reference_base64_image}"
        }
    })
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{result_base64_image}"
        }
    })
    message = [{
        "role": "user",
        "content": content
    }]

    evaluation, usage = invoke_vision(message)
    print(' Evaluator Answer '.center(80, '-'))
    print(evaluation)
    print(usage)
    print()

    # Example of teh eevaluation: "score" : [10, 9], "reasoning" : "The left bird in the original image (a hornbill) has been successfully changed to a chicken-like bird in the edited image, accurately following the instruction. The rest of the scene, including lighting, pose, and the right bird, remains nearly identical, showing minimal overediting."
    # Parse the evaluation
    evaluation_json_str = '{' + evaluation + '}'
    evaluation_json = json.loads(evaluation_json_str)
    score = evaluation_json['score']
    reasoning = evaluation_json['reasoning']

    return score, reasoning 



def evaluate_t2i(task: dict) -> bool:
    result_base64_image, result_meta_info = load_image(task['result'])

    prompt = t2i_prompt.format(
        instruction=task['instruction'],
        result_resolution=f'{result_meta_info["width"]}x{result_meta_info["height"]}'
    )
    logger.info(' Evaluator Prompt '.center(80, '-'))
    logger.info(prompt)
    logger.info()

    content = [{
        'type': 'text',
        'text': prompt
    }]
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{result_base64_image}"
        }
    })
    message = [{
        "role": "user",
        "content": content
    }]
    evaluation, usage = invoke_vision(message)
    logger.info(' Evaluator Answer '.center(80, '-'))
    logger.info(evaluation)
    logger.info(usage)
    logger.info()

    analysis, judgment = parse_evaluation(evaluation)
    logger.info(' Parsed Analysis '.center(80, '-'))
    logger.info(analysis)
    logger.info()
    logger.info(' Parsed Judgment '.center(80, '-'))
    logger.info(judgment)
    logger.info()

    return analysis, judgment


def evaluate_i2i(task: dict) -> bool:
    if task['resource'].startswith('/'): # Absolute path
        reference_base64_image, reference_meta_info = load_image(f'{task["resource"]}')
    else: # Relative path
        reference_base64_image, reference_meta_info = load_image(f'./dataset/benchmark/resource/{task["resource"]}')
    result_base64_image, result_meta_info = load_image(f'{task["result"]}')

    prompt = i2i_prompt.format(
        instruction=task['instruction'],
        reference_resolution=f'{reference_meta_info["width"]}x{reference_meta_info["height"]}',
        result_resolution=f'{result_meta_info["width"]}x{result_meta_info["height"]}'
    )
    logger.info(' Evaluator Prompt '.center(80, '-'))
    logger.info(prompt)
    logger.info()

    content = [{
        'type': 'text',
        'text': prompt
    }]
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{reference_base64_image}"
        }
    })
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{result_base64_image}"
        }
    })
    message = [{
        "role": "user",
        "content": content
    }]
    evaluation, usage = invoke_vision(message)
    logger.info(' Evaluator Answer '.center(80, '-'))
    logger.info(evaluation)
    logger.info(usage)
    logger.info()

    analysis, judgment = parse_evaluation(evaluation)
    logger.info(' Parsed Analysis '.center(80, '-'))
    logger.info(analysis)
    logger.info()
    logger.info(' Parsed Judgment '.center(80, '-'))
    logger.info(judgment)
    logger.info()

    return analysis, judgment


def evaluate_t2v(task: dict) -> bool:
    result_base64_frames, result_meta_info = load_video(task['result'])

    prompt = t2v_prompt.format(
        instruction=task['instruction'],
        result_frame_count=len(result_base64_frames),
        result_resolution=f'{result_meta_info["width"]}x{result_meta_info["height"]}',
        result_duration=f'{result_meta_info["duration"]:.2f}',
        result_frame_rate=f'{result_meta_info["frame_rate"]}'
    )
    logger.info(' Evaluator Prompt '.center(80, '-'))
    logger.info(prompt)
    logger.info()

    content = [{
        'type': 'text',
        'text': prompt
    }]
    for result_base64_frame in result_base64_frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{result_base64_frame}"
            }
        })
    message = [{
        "role": "user",
        "content": content
    }]
    evaluation, usage = invoke_vision(message)
    logger.info(' Evaluator Answer '.center(80, '-'))
    logger.info(evaluation)
    logger.info(usage)
    logger.info()

    analysis, judgment = parse_evaluation(evaluation)
    logger.info(' Parsed Analysis '.center(80, '-'))
    logger.info(analysis)
    logger.info()
    logger.info(' Parsed Judgment '.center(80, '-'))
    logger.info(judgment)
    logger.info()

    return analysis, judgment


def evaluate_i2v(task: dict) -> bool:
    reference_base64_image, reference_meta_info = load_image(f'./dataset/benchmark/resource/{task["resource"]}')
    result_base64_frames, result_meta_info = load_video(task['result'])

    prompt = i2v_prompt.format(
        instruction=task['instruction'],
        reference_resolution=f'{reference_meta_info["width"]}x{reference_meta_info["height"]}',
        result_frame_count=len(result_base64_frames),
        result_resolution=f'{result_meta_info["width"]}x{result_meta_info["height"]}',
        result_duration=f'{result_meta_info["duration"]:.2f}',
        result_frame_rate=f'{result_meta_info["frame_rate"]}'
    )
    logger.info(' Evaluator Prompt '.center(80, '-'))
    logger.info(prompt)
    logger.info()

    content = [{
        'type': 'text',
        'text': prompt
    }]
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{reference_base64_image}"
        }
    })
    for result_base64_frame in result_base64_frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{result_base64_frame}"
            }
        })
    message = [{
        "role": "user",
        "content": content
    }]
    evaluation, usage = invoke_vision(message)
    logger.info(' Evaluator Answer '.center(80, '-'))
    logger.info(evaluation)
    logger.info(usage)
    logger.info()

    analysis, judgment = parse_evaluation(evaluation)
    logger.info(' Parsed Analysis '.center(80, '-'))
    logger.info(analysis)
    logger.info()
    logger.info(' Parsed Judgment '.center(80, '-'))
    logger.info(judgment)
    logger.info()

    return analysis, judgment


def evaluate_v2v(task: dict) -> bool:
    reference_base64_frames, reference_meta_info = load_video(f'./dataset/benchmark/resource/{task["resource"]}')
    result_base64_frames, result_meta_info = load_video(task['result'])

    prompt = v2v_prompt.format(
        instruction=task['instruction'],
        reference_frame_count=len(reference_base64_frames),
        reference_resolution=f'{reference_meta_info["width"]}x{reference_meta_info["height"]}',
        reference_duration=f'{reference_meta_info["duration"]:.2f}',
        reference_frame_rate=f'{reference_meta_info["frame_rate"]}',
        result_frame_count=len(result_base64_frames),
        result_resolution=f'{result_meta_info["width"]}x{result_meta_info["height"]}',
        result_duration=f'{result_meta_info["duration"]:.2f}',
        result_frame_rate=f'{result_meta_info["frame_rate"]}'
    )
    logger.info(' Evaluator Prompt '.center(80, '-'))
    logger.info(prompt)
    logger.info()

    content = [{
        'type': 'text',
        'text': prompt
    }]
    for reference_base64_frame in reference_base64_frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{reference_base64_frame}"
            }
        })
    for result_base64_frame in result_base64_frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{result_base64_frame}"
            }
        })
    message = [{
        "role": "user",
        "content": content
    }]
    evaluation, usage = invoke_vision(message)
    logger.info(' Evaluator Answer '.center(80, '-'))
    logger.info(evaluation)
    logger.info(usage)
    logger.info()

    analysis, judgment = parse_evaluation(evaluation)
    logger.info(' Parsed Analysis '.center(80, '-'))
    logger.info(analysis)
    logger.info()
    logger.info(' Parsed Judgment '.center(80, '-'))
    logger.info(judgment)
    logger.info()

    return analysis, judgment


def evaluate_reason_t2i(task: dict) -> bool:
    result_base64_image, result_meta_info = load_image(task['result'])

    prompt = t2i_prompt_reasoning.format(
            instruction=task['instruction'],
            result_resolution=f'{result_meta_info["width"]}x{result_meta_info["height"]}'
    )
    logger.info(' Evaluator Prompt '.center(80, '-'))
    logger.info(prompt)
    logger.info()

    content = [{
        'type': 'text',
        'text': prompt
    }]
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{result_base64_image}"
        }
    })
    message = [{
        "role": "user",
        "content": content
    }]
    evaluation, usage = invoke_vision(message)
    logger.info(' Evaluator Answer '.center(80, '-'))
    logger.info(evaluation)
    logger.info(usage)
    logger.info()

    analysis, judgment = parse_evaluation(evaluation)
    logger.info(' Parsed Analysis '.center(80, '-'))
    logger.info(analysis)
    logger.info()
    logger.info(' Parsed Judgment '.center(80, '-'))
    logger.info(judgment)
    logger.info()

    return analysis, judgment





