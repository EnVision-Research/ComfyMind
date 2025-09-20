import os
import sys
import time
import json
import argparse
import logging
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.ComfyMind_System import ComfyMind
from script.evaluation_geneval import evaluate_geneval_image

import argparse

def main():
    # nohup python eval_geneval.py &
    parser = argparse.ArgumentParser(description='Run and eval in Benchmark geneval.')
    parser.add_argument('--log_path', type=str, default='results/geneval/log.log', help='The path to save the log file.')
    parser.add_argument('--dataset_path', type=str, default='dataset/geneval', help='The path to the dataset.')
    parser.add_argument('--result_json_path', type=str, default='results/geneval/result.json', help='The path to the result json file.')
    parser.add_argument('--output_path', type=str, default='results/geneval/output', help='The path to the output image.')
    parser.add_argument('--skip_exist', type=bool, default=True, help='Whether to skip the existing tasks.')
    parser.add_argument('--workflow_meta_info', type=str, default='atomic_workflow/meta_doc/Meta_Info_Geneval.json', help='The path to the workflow meta info.')
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

    # 使用ComfyMind_System，设置eval_agent为'geneval'
    pipeline = ComfyMind(
        eval_agent='geneval', 
        meta_info_site=args.workflow_meta_info,
        preprocessing=None
    )

    # prepare the task data
    dataset_All = []
    file_path = "dataset/geneval/evaluation_metadata.jsonl" 
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
                        if identifier in result_json_all and args.skip_exist == True :
                            print(f"Task {identifier} already exists in the result_json_path, skip it")
                            continue
                    except json.JSONDecodeError:
                        print(f"Error: {e}")
                        logging.info(f"Error: {e}")
            else:
                result_json_all = {}  # 文件不存在或为空，初始化为空字典

            instruction = task["prompt"]
                
            # 传入 pipeline - 使用ComfyMind_System的调用方式
            print(f'Finishing data and instruction preparation...')
            print(f'Start to run pipeline...')
            
            # 构造ComfyMind_System需要的task格式
            comfymind_task = {
                'instruction': f'Instruction: {instruction}',
                'resource1': None,
                'resource2': None
            }
            
            result = pipeline(comfymind_task)
            print(f"result: {result}")

            # 处理ComfyMind_System的返回结果
            if result['status'] == 'completed':
                # 获取生成的图像路径
                image_path = result['output']
                
                # 使用Geneval评估
                result_json = evaluate_geneval_image(image_path=image_path, metadata=task)
                print(f"result_json: {result_json}")
                
                new_filename = f"{i}.png"
                target_path = os.path.join(args.output_path, new_filename)
                # transfer the image to 'eval/geneval/output'
                import shutil
                shutil.copy(image_path, target_path)

                # update result_json
                result_json['filename'] = target_path
                print(f"result_json Updated: {result_json}")
                print(f"Whether Success: {result_json['correct']}")

                result_json_all[identifier] = result_json
                with open(args.result_json_path, 'w', encoding='utf-8') as file:
                    json.dump(result_json_all, file, ensure_ascii=False, indent=4)
            else:
                # 处理失败情况
                print(f"Task failed: {result['error_message']}")
                logging.info(f"Task failed: {result['error_message']}")
                
                # 创建失败的结果记录
                result_json = {
                    'correct': False,
                    'reason': result['error_message'],
                    'filename': None,
                    'tag': task.get('tag', 'unknown')
                }
                
                result_json_all[identifier] = result_json
                with open(args.result_json_path, 'w', encoding='utf-8') as file:
                    json.dump(result_json_all, file, ensure_ascii=False, indent=4)
                    
        except Exception as e:
            print(f"Error: {e}")
            logging.info(f"Error: {e}")
            continue


if __name__ == '__main__':
    main()

