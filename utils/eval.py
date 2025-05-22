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
from utils.model import invoke_text, invoke_vision
from utils.eval_utils import evaluate_t2i, evaluate_i2i, evaluate_t2v, evaluate_i2v, evaluate_v2v, evaluate_reason_t2i

def check_generation_judgment(output, task, reasoning: bool = False, skip_judgment: bool = False):

    result = output['outputs']
    logger.info(f'result: {result}')
    logger.info(f'type of result: {type(result)}')
    logger.info(f'len of result: {len(result)}')

    # Integrate result into task
    modality = task['modality']
    result_path = None

    for item in result:
        logger.info(f'item: {item}')
        logger.info(f'type of item: {type(item)}')
        if modality in ['t2i', 'i2i', 'reasoningt2i'] and 'images' in item:
            task['result'] = item['images']
            result_path = item['images']
        elif modality in ['t2v', 'i2v', 'v2v'] and 'videos' in item:
            task['result'] = item['videos']
            result_path = item['videos']

    if skip_judgment:
        judgment = 'True'
        analysis = 'Skip Judgment'
        return analysis, judgment, result_path


    logger.info(f'task: {task}')
    # Evaluate the generated result
    evaluation_functions = {
        't2i': evaluate_t2i,
        'i2i': evaluate_i2i,
        't2v': evaluate_t2v,
        'i2v': evaluate_i2v,
        'v2v': evaluate_v2v,
        'reasoningt2i': evaluate_reason_t2i,
    }
    
    if modality in evaluation_functions:
        analysis, judgment = evaluation_functions[modality](task)
    else:
        raise RuntimeError(f'Invalid modality: {modality}')

    # Output the evaluation results
    logger.info(f'Evaluation Analysis: {analysis}')
    logger.info(f'Evaluation Judgment: {judgment}')
    logger.info("End of The Self Evaluation".center(80, '-'))

    return analysis, judgment, result_path



