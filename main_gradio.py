import os
import logging
import shutil
import cv2
import numpy as np
from PIL import Image
import io
import datetime

import gradio as gr

from agent.ComfyMind_System import ComfyMind
from utils.model import invoke_text

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
LOG_PATH = 'log/log.log'
RESOURCES_DIR = 'resources'
META_INFO_SITE = 'atomic_workflow/meta_doc/Meta_Info.json'

# Ensure directories exist
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(RESOURCES_DIR, exist_ok=True)

# Basic logging to file + console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_PATH, mode='w'),
        logging.StreamHandler()
    ]
)


def judge_modality(task):
    message = [
        {"role": "system", "content": """
          You are a helpful assistant that judges the modality of the task according to the instruction and the input visual resource.
          The output modality should be one of the following: t2i, i2i, t2v, i2v, v2v, reasont2i, NotSupported.
          T2I: Text to Image Generation
          I2I: Image to Image Generation
          T2V: Text to Video Generation
          I2V: Image to Video Generation
          V2V: Video to Video Generation
          ReasonT2I: Reasoning Text to Image
          NotSupported: Not Supported (Such as audio, 3D, etc.)
          And it should be wrapped in <modality> </modality> tags.
         """},
        {"role": "user", "content": f"Task: {task}"}
    ]
    response = invoke_text(message)
    modality = response.split('<modality>')[1].split('</modality>')[0]
    return modality

# -----------------------------------------------------------------------------
# Core execution wrapper
# -----------------------------------------------------------------------------

def file_to_path(file_data):
    """Convert Gradio FileData to file path, Convert into absolute path"""
    if file_data is None:
        return None
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = os.path.splitext(file_data.name)[1] if hasattr(file_data, 'name') else '.png'
    filename = f"input_{timestamp}{file_extension}"
    filepath = os.path.join(RESOURCES_DIR, filename)
    
    if isinstance(file_data, str):
        shutil.copy2(file_data, filepath)
    else:
        try:
            if hasattr(file_data, 'read'):
                with open(filepath, 'wb') as f:
                    f.write(file_data.read())
            else:
                logging.error(f"Unsupported file data type: {type(file_data)}")
                return None
        except Exception as e:
            logging.error(f"Error saving file: {str(e)}")
            return None
    
    return os.path.abspath(filepath)


def run_task(instruction: str,
             resource1,
             resource2,
             meta_info_site: str = META_INFO_SITE):
    """Run a single ComfyMind task and return the resulting file path."""

    pipeline = ComfyMind(
        meta_info_site=meta_info_site,
        preprocessing='prompt_optimization'
    )

    resource1_path = file_to_path(resource1)
    resource2_path = file_to_path(resource2)

    task_name = 'task'

    task = {
        'name': task_name,
        'instruction': f'Instruction: {instruction}, and input visual resource: {resource1_path} and {resource2_path}',
        'resource1': resource1_path,
        'resource2': resource2_path,
        'modality': '',
    }

    modality = judge_modality(task)

    if modality not in ['t2i', 'i2i', 't2v', 'i2v', 'v2v', 'reasont2i']:
        logging.info(f'Modality: {modality}')
        return f'Error: NotSupported modality', None, None

    task['modality'] = modality
    logging.info(f'Modality: {modality}')

    logging.info(f'Running task: {task}')
    try:
        result = pipeline(task)
    except Exception as e:
        logging.error(f"Error running task: {str(e)}")
        message = result['error_message']
        return message, None, None

    if result['status'] != 'completed':
        return f'{result["status"]}: {result["error_message"]}', None, None

    if modality in ['t2v', 'i2v', 'v2v']:
        final_path = os.path.join(RESOURCES_DIR, f'{task_name}.mp4')
        shutil.copy2(result['output'], final_path)
        return None, None, final_path
    else:
        final_path = os.path.join(RESOURCES_DIR, f'{task_name}.png')
        shutil.copy2(result['output'], final_path)
        return None, final_path, None


def main():
    with gr.Blocks(css="""
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 0;
        }
        #col-container { margin: 0px auto; max-width: 1200px; } 
        .gradio-container {
            max-width: 1200px;
            margin: auto;
            width: 100%;
        }
        #center-align-column {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        h1 {text-align: center;}
        h2 {text-align: center;}
        .center {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 50%;
        }
    """, elem_id="col-container") as interface:
        with gr.Row(elem_id="content-container"):
            with gr.Column(scale=7, elem_id="center-align-column"):
                gr.Markdown("""
                <div style="text-align: center;">
                    <h1 style="font-size: 2.5em; margin-bottom: 0.5em;">Official Live Demo</h1>
                    <h2 style="font-size: 1.8em; margin-bottom: 1em; font-weight: 600;">ComfyMind: Toward General-Purpose Generation via Tree-Based Planning and Reactive Feedback</h2>
                    <div style="display: flex; justify-content: center; align-items: center; gap: 10px; margin-bottom: 2em;">
                        <a href="https://arxiv.org/abs/xxxx.xxxxx" target="_blank">
                            <img src="https://img.shields.io/badge/arXiv-Link-red" alt="arXiv">
                        </a>
                        <a href="https://github.com" target="_blank">
                            <img src="https://img.shields.io/badge/GitHub-Repo-blue" alt="GitHub">
                        </a>
                    </div>
                </div>
                """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### Input Section
                Enter your instruction and upload reference files here.
                """)
                instruction = gr.Textbox(
                    label="Instruction", 
                    lines=2,
                    placeholder="Enter generation instruction here. For example: 'Generate an image of a cat playing with a ball'"
                )
                resource1 = gr.File(
                    label="Reference Image/Video 1 (optional)",
                    file_types=["image", "video"],
                    file_count="single",
                    type="filepath"
                )
                resource2 = gr.File(
                    label="Reference Image/Video 2 (optional)",
                    file_types=["image", "video"],
                    file_count="single",
                    type="filepath"
                )
            
            with gr.Column():
                gr.Markdown("""
                ### Output Section
                Generated results will appear here.
                """)
                message = gr.Textbox(
                    label="Message",
                    lines=2,
                    placeholder="Status messages will appear here",
                    interactive=False
                )
                output_image = gr.Image(
                    label="Output Image",
                    type="filepath",
                    interactive=False
                )
                output_video = gr.Video(
                    label="Output Video",
                    interactive=False,
                    autoplay=True
                )
        
        submit_btn = gr.Button("Generate", variant="primary")
        submit_btn.click(
            fn=run_task,
            inputs=[instruction, resource1, resource2],
            outputs=[message, output_image, output_video]
        )
    
    interface.launch(server_port=8888, server_name="0.0.0.0", share=False)

if __name__ == '__main__':
    main()


# tmux new -s gradio_app
# python main_gradio.py
# # 按 Ctrl+B 然后按 D 来分离tmux会话
# lsof -i :6116

# tmux new -s comfymind
# python main.py