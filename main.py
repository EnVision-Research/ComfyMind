import os
import sys
import time
import json
import argparse
import logging
import yaml

from agent.ComfyMind_System import ComfyMind
from main_gradio import judge_modality
import argparse

def main():
    # python run_task.py --task_name task_1 --instruction "Generation a image of beautiful future city" 

    parser = argparse.ArgumentParser(description='Run A Task.')

    parser.add_argument('--log_path', type=str, default='log/log.log', help='The path to save the log file.')
    parser.add_argument('--dataset_path', type=str, default='dataset/comfybench', help='The path to the dataset.')
    parser.add_argument('--output_path', type=str, default='outputs/', help='The path to the output image.')
    parser.add_argument('--eval_agent', type=str, default='normal', help=" 'normal' , 'geneval' or 'none' ")
    parser.add_argument('--meta_info_site', type=str, default='atomic_workflow/meta_doc/Meta_Info.json', help='The path to the meta info (description of the atomic workflow) file.')
    parser.add_argument('--preprocessing', type=str, default='prompt_optimization', help='how to preprocess the task, including none, prompt_optimization, instruction_analysis.') 

    parser.add_argument('--task_name', type=str, default='task', help='The name of the task.')
    parser.add_argument('--instruction', type=str, default='', help='The instruction of the task.')
    parser.add_argument('--resource1', type=str, default='', help='The path to the input resource image/video.')
    parser.add_argument('--resource2', type=str, default='', help='The path to the input resource image/video.')
    parser.add_argument('--task_modality', type=str, default='t2i', help='The modality of the task including t2i, i2i, t2v, i2v, v2v, reasont2i.')
    
    args = parser.parse_args()

    # check log and output path
    if not os.path.exists(args.log_path):
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    if not os.path.exists(args.output_path):
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Setup file and console logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(args.log_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    logging.info(' Program Status '.center(80, '-'))
    logging.info('creating pipeline...')

    pipeline = ComfyMind(eval_agent=args.eval_agent, meta_info_site=args.meta_info_site, preprocessing=args.preprocessing)

    resource1_path = os.path.abspath(args.resource1)
    resource2_path = os.path.abspath(args.resource2)

    task = {
        'name': args.task_name,
        'instruction': f'Instruction: {args.instruction}, and input visual resource: {resource1_path} and {resource2_path}',
        'resource1': resource1_path,
        'resource2': resource2_path,
        'modality': '',
    }
    modality = judge_modality(task)
    if modality not in ['t2i', 'i2i', 't2v', 'i2v', 'v2v', 'reasont2i']:
        logging.info(f'Modality: {modality}')
        raise ValueError(f'Modality: {modality} is not supported')
    task['modality'] = modality

    try :

        result = pipeline(task)
        output_path = result['output']

        print(' Task Finished '.center(80, '-'))

        import shutil
        output_path = os.path.join(args.output_path, f'{args.task_name}.png')
        shutil.copy2(result['output'], output_path)

        logging.info(f"result: {output_path}")

    except Exception as e:
        logging.info(f"Error: {e}")

if __name__ == '__main__':
    main()

