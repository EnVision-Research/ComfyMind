import os
import sys
import time
import json
import argparse
import logging
import yaml


from utils import console
from utils.comfy import execute_prompt
from agent.zeroshot import ZeroShotPipeline
from agent.Heuristic_Search_Reason_Edit import Heuristic_Search_Reason_Edit
# from agent.Heuristic_Search_All import Heuristic_Search_All
from script.evaluation import evaluate_reason_edit  # Import evaluation functions


import argparse

def main():
    # nohup python eval_reason_edit.py > logeval.out 2>&1 &
    parser = argparse.ArgumentParser(description='Run and eval in Benchmark Reason_Edit.')
    parser.add_argument('--log_path', type=str, default='eval/reason_edit/log.log', help='The path to save the log file.')
    parser.add_argument('--dataset_path', type=str, default='dataset/reason_edit', help='The path to the dataset.')
    parser.add_argument('--result_json_path', type=str, default='eval/reason_edit/result.json', help='The path to the result json file.')
    parser.add_argument('--output_path', type=str, default='eval/reason_edit/output', help='The path to the output image.')
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

    pipeline = Heuristic_Search_Reason_Edit(workflow_folder_path='workflow_reason_edit')

    task__dic_list = ['1-Left-Right', '2-Relative-Size', '3-Mirror', '4-Color', '5-Multiple-Objects', '6-Reasoning', '7-Add-supp']
    task_file_name = ['Left_Right.txt', 'Size_text.txt', 'Mirror_text.txt', 'Color_text.txt', 'MultipleObjects_text.txt', 'Reason_test.txt', 'add_text.txt']
    for i in range(len(task__dic_list)):
        task_file_path = os.path.join(args.dataset_path, task__dic_list[i], task_file_name[i])
        # Read by line, and record it

        instruction_list = []
        with open(task_file_path, 'r') as file:
            for line in file:
                instruction_list.append(line.strip())
        
        # avali_list = []
        #avali_list = [2, 3, 6, 8, 10, 12, 15, 16, 18, 19, 20, 21, 22]
        for number_, instruction in enumerate(instruction_list):
            number = number_ + 1

            #if tot_task_num != 0:
            #    print(f'Average score: {tot_score / tot_task_num}')


            print(f"------------------------------- Start to process task {number} -------------------------------")
            logging.info(f"------------------------------- Start to process task {number} -------------------------------")
            print(f"instruction: {instruction}")
            logging.info(f"instruction: {instruction}")

            instruction_id = f"{number:03}"  # Format ID as '001', '002', ..., '200'
            identifier = f"{task__dic_list[i]}_{instruction_id}"
            
            # Check if the instruction_id is not exist in the result_json_path
            if os.path.exists(args.result_json_path) and os.path.getsize(args.result_json_path) > 0:
                with open(args.result_json_path, 'r', encoding='utf-8') as file:
                    try:
                        result_json = json.load(file)  # 读取已有 JSON 数据
                        if identifier in result_json and args.skip_exist == True:
                            print(f"Task {identifier} already exists in the result_json_path, skip it")
                            tot_score += result_json[identifier]['score'][2]
                            tot_task_num += 1
                            print(f'Average score: {tot_score / tot_task_num}')
                            continue
                    except json.JSONDecodeError:
                        print(f"Error: {e}")
                        logging.info(f"Error: {e}")
            else:
                result_json = {}  # 文件不存在或为空，初始化为空字典

            try:
                # check .png or .jpg
                image_paths = f'/remote-home/share/litaoguo/ComfyBench/{args.dataset_path}/{task__dic_list[i]}/{instruction_id}.png' if os.path.exists(f'{args.dataset_path}/{task__dic_list[i]}/{instruction_id}.png') else f'{args.dataset_path}/{task__dic_list[i]}/{instruction_id}.jpg'
            except Exception as e:
                print(f"Error: {e}")
                logging.info(f"Error: {e}")
                image_paths = None
            
            # 定位'CLIP:'的位置，其前面为instruction, 没找到则返回原instruction
            clip_index = instruction.find('CLIP:')
            instruction = instruction[:clip_index] if clip_index != -1 else instruction

            # 获取指定编号的内容
            task = {}
            task['name'] = identifier
            task['instruction'] = f'Instruction: {instruction}, reference image: {image_paths}'
            task['input_image'] = image_paths
            task['score'] = []
            task['output_image'] = None
            task['modality'] = 'i2i'
            task['resource'] = image_paths
            task['category'] = 'Reason_Edit'
            
            # 传入 pipeline
            print(f'Finishing data and instruction preparation...')
            print(f'Start to run pipeline...')

            print(f'task pass to pipeline: {task}')
            
            result = pipeline(f'Instruction: {instruction}, reference image: {image_paths}', task, image_paths=None)
            print(f"result: {result}")
            task['output_image'] = result['output']
            #task['output_image'] = '/remote-home/share/litaoguo/ComfyBench/outputs/images/ComfyUI_temp_idhzr_00001_.png'

            if task['output_image'] is None:
                print(f'Task {identifier} failed, No output image')
                logging.info(f'Task {identifier} failed, No output image')
                continue

            # Ensure the output directory exists
            output_path = args.output_path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Save the output to output path
            output_path = os.path.join(args.output_path, f'{identifier}.png')
            
            # Simply copy the file to the destination
            import shutil
            shutil.copy2(task['output_image'], output_path)
            task['output_image'] = output_path

            score, reasoning = evaluate_reason_edit(task=task)
            min_score = min(score)
            score.append(min_score)

            # update the image path
            task['output_image'] = output_path
            # update the score and reasoning
            task['score'] = score
            task['reasoning'] = reasoning

            # print and log the final task status
            print(f'------------------------------- Score: {score} -------------------------------')
            logging.info(f'------------------------------- Score: {score} -------------------------------')


            result_json[identifier] = task
            tot_score += result_json[identifier]['score'][2]
            tot_task_num += 1
            print(f'Average score: {tot_score / tot_task_num}')

            # save the task to the result_json_path
            with open(args.result_json_path, 'w', encoding='utf-8') as file:
                json.dump(result_json, file, ensure_ascii=False, indent=4)
            



if __name__ == '__main__':
    main()

