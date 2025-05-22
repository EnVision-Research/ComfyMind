import os
import sys
import time
import json
import argparse
import logging
import yaml


from utils import console
from utils.comfy import execute_prompt
from agent.Heuristic_Search_Geneval import Heuristic_Search_Geneval


import argparse

def main():
    # nohup python eval_geneval.py &
    parser = argparse.ArgumentParser(description='Run and eval in Benchmark geneval.')
    parser.add_argument('--log_path', type=str, default='eval/geneval/log.log', help='The path to save the log file.')
    parser.add_argument('--dataset_path', type=str, default='dataset/geneval', help='The path to the dataset.')
    parser.add_argument('--result_json_path', type=str, default='eval/geneval/result.json', help='The path to the result json file.')
    parser.add_argument('--output_path', type=str, default='eval/geneval/output', help='The path to the output image.')
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

    pipeline = Heuristic_Search_Geneval(workflow_folder_path='workflow_geneval')

    # prepare the task data
    dataset_All = []
    file_path = "dataset/geneval/evaluation_metadata.jsonl"  # 替换为你的文件路径
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                dataset_All.append(data)
            except json.JSONDecodeError as e:
                print(f"Geneval JSON error: {e}")

    for i in range(len(dataset_All)):

        try :
            identifier = f'{i}'
            task = dataset_All[i]

            print(f"------------------------------- Start to process task {i} -------------------------------")
            logging.info(f"------------------------------- Start to process task {i} -------------------------------")

            print(f"task: {task}")
            logging.info(f"task: {task}")
            
                
            # Check if the instruction_id is not exist in the result_json_path
            if os.path.exists(args.result_json_path) and os.path.getsize(args.result_json_path) > 0:
                with open(args.result_json_path, 'r', encoding='utf-8') as file:
                    try:
                        result_json_all = json.load(file)  # 读取已有 JSON 数据
                        if identifier in result_json_all and args.skip_exist == True and identifier != '289' : # ensure the prompt is the same
                            print(f"Task {identifier} already exists in the result_json_path, skip it")
                            continue
                    except json.JSONDecodeError:
                        print(f"Error: {e}")
                        logging.info(f"Error: {e}")
            else:
                result_json_all = {}  # 文件不存在或为空，初始化为空字典

            # E.g. of task : {"tag": "single_object", "include": [{"class": "refrigerator", "count": 1}], "prompt": "a photo of a refrigerator"}

            instruction = task["prompt"]
                
            # 传入 pipeline
            print(f'Finishing data and instruction preparation...')
            print(f'Start to run pipeline...')
            
            result = pipeline(f'Instruction: {instruction}', task, image_paths=None)
            print(f"result: {result}")

            result_json = result['output']
            print(f"result_json: {result_json}")

            new_filename = f"{i}.png"
            target_path = os.path.join(args.output_path, new_filename)
            # transfer the image to 'eval/geneval/output'
            image_file_path = result_json['filename']
            import shutil
            shutil.copy(image_file_path, target_path)

            # update result_json
            result_json['filename'] = target_path
            print(f"result_json Updated: {result_json}")
            print(f"Whether Success: {result_json['correct']}")

            result_json_all[identifier] = result_json
            with open(args.result_json_path, 'w', encoding='utf-8') as file:
                json.dump(result_json_all, file, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error: {e}")
            logging.info(f"Error: {e}")
            continue



    with open(args.result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 按 key（字符串形式的数字）排序
    sorted_data = dict(sorted(data.items(), key=lambda item: int(item[0])))

    # 统计每个类别的正确率
    tag_all = {}
    for i in range(len(dataset_All)):
        task = dataset_All[i]
        tag = task['tag']
        if tag not in tag_all:
            tag_all[tag] = {'Total': 0, 'Correct': 0, 'Incorrect': 0, 'Accuracy': 0, 'Missing': 0}
        tag_all[tag]['Total'] += 1

    for key, value in sorted_data.items():
        if value['correct'] == True:
            tag_all[value['tag']]['Correct'] += 1
        else:
            tag_all[value['tag']]['Incorrect'] += 1

    for key, value in tag_all.items():
        value['Accuracy'] = value['Correct'] / value['Total']
        value['Missing'] = value['Total'] - value['Correct'] - value['Incorrect']

    # Save as another json file
    with open(args.result_json_path.replace(".json", "_sorted.json"), "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=4)

    with open(args.result_json_path.replace(".json", "_stat.json"), "w", encoding="utf-8") as f:
        json.dump(tag_all, f, ensure_ascii=False, indent=4)

    print(f"tag_all: {tag_all}")

if __name__ == '__main__':
    main()

