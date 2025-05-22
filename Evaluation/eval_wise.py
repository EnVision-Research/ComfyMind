import os
import sys
import time
import json
import argparse
import logging
import yaml


from utils import console
from utils.comfy import execute_prompt
from agent.Heuristic_Search_Wise import Heuristic_Search_Wise
from script.evaluation_wise import evaluate_single_image_wise

import argparse

def main():
    # nohup python eval_wise.py &
    # and (result_json_all[identifier]['Category'] != 'Biology' or result_json_all[identifier]['scores']['consistency'] > 1.0)
    parser = argparse.ArgumentParser(description='Run and eval in Benchmark wise.')
    parser.add_argument('--log_path', type=str, default='eval/wise/log.log', help='The path to save the log file.')
    parser.add_argument('--dataset_path', type=str, default='dataset/wise', help='The path to the dataset.')
    parser.add_argument('--result_json_path', type=str, default='eval/wise/result.json', help='The path to the result json file.')
    parser.add_argument('--output_path', type=str, default='eval/wise/output', help='The path to the output image.')
    parser.add_argument('--skip_exist', type=bool, default=True, help='Whether to skip the existing tasks.')
    args = parser.parse_args()

    logging.basicConfig(
        filename=f'{args.log_path}', 
        level=logging.INFO
    )
    
    tot_score = 0
    tot_task_num = 0

    # create pipeline
    print(' Program Status '.center(80, '-'))
    print('creating pipeline...')
    print()

    # pipeline = Heuristic_Search_Wise(workflow_folder_path='workflow_wise')
    pipeline = None

    # prepare the task data
    dataset_All = []
    file_path_cultural = "dataset/wise/cultural_common_sense.json" 
    file_path_time = "dataset/wise/spatio-temporal_reasoning.json"
    file_path_science = "dataset/wise/natural_science.json"
    dataset_json_file = [file_path_cultural, file_path_time, file_path_science]

    dataset_All = []
    for file in dataset_json_file:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            dataset_All.extend(data)

    total_score = 0
    total_task_num = 0

    category_all = {}
    for i in range(len(dataset_All)):
        try :
            task = dataset_All[i]
            identifier = f'{task["prompt_id"]}'

            print(f"------------------------------- Start to process task {identifier} -------------------------------")
            logging.info(f"------------------------------- Start to process task {identifier} -------------------------------")

            print(f"task: {task}")
            logging.info(f"task: {task}")
            
                
            # Check if the instruction_id is not exist in the result_json_path
            if os.path.exists(args.result_json_path) and os.path.getsize(args.result_json_path) > 0:
                with open(args.result_json_path, 'r', encoding='utf-8') as file:
                    try:
                        result_json_all = json.load(file)  # 读取已有 JSON 数据
                        if args.skip_exist == True and identifier in result_json_all : # skip the existing tasks
                            if task['Category'] not in category_all:
                                category_all[task['Category']] = {}
                                category_all[task['Category']]['num'] = 0
                                category_all[task['Category']]['score'] = 0
                            
                            category_all[task['Category']]['num'] += 1
                            category_all[task['Category']]['score'] += result_json_all[identifier]['scores']['normalized_mean']

                            total_task_num += 1
                            total_score += result_json_all[identifier]['scores']['normalized_mean']
                            print(f"average score: {total_score / total_task_num}")
                            continue
                    except json.JSONDecodeError:
                        print(f"Error: {e}")
                        logging.info(f"Error: {e}")
            else:
                result_json_all = {}  # 文件不存在或为空，初始化为空字典

            # E.g. of task : {"tag": "single_object", "include": [{"class": "refrigerator", "count": 1}], "prompt": "a photo of a refrigerator"}

            instruction = task["Prompt"]
            explanation = task["Explanation"]
            
            instruction_input = f'Instruction: Generate an image of: {instruction}'
            # 传入 pipeline
            print(f'Finishing data and instruction preparation...')
            print(f'Start to run pipeline...')
            
            task_temp_for_eval = {
                'name': identifier,
                'instruction': instruction_input,
                'output_image': None,
                'modality': 't2i',
                'resource': None
            }

            result = pipeline(instruction_input, task_temp_for_eval, image_paths=None)
            image_path = result['output']
            print(f"result: {image_path}")

            scores = evaluate_single_image_wise(image_path=image_path, prompt_data={"Prompt": instruction, "Explanation": explanation})

            print(f"scores: {scores}")

            task['scores'] = scores
            result_json_all[identifier] = task

            # Resave the new generated image
            import shutil
            output_path = os.path.join(args.output_path, f'{identifier}.png')
            shutil.copy2(image_path, output_path)
            task['output_image'] = output_path

            with open(args.result_json_path, 'w', encoding='utf-8') as file:
                json.dump(result_json_all, file, ensure_ascii=False, indent=4)

            total_task_num += 1
            total_score += scores['normalized_mean']
            print(f"average score: {total_score / total_task_num}")

        except Exception as e:
            print(f"Error: {e}")
            logging.info(f"Error: {e}")
            continue

    
    for item in category_all:
        print(f"category: {item}, num: {category_all[item]['num']}, score: {category_all[item]['score'] / category_all[item]['num']}")


if __name__ == '__main__':
    main()

