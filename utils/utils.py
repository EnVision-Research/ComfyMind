from PIL import Image
import io
import base64
import json
import logging
import os
logger = logging.getLogger(__name__) 
from utils.tools import FUNCTIONS, TOOLS
from utils.model import invoke_text

def encode_image(image: Image.Image, size: tuple[int, int] = (512, 512)) -> str:
    image.thumbnail(size)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_image

def load_image(image_path: str, size_limit: tuple[int, int] = (512, 512)) -> str:
    image = Image.open(image_path)
    base64_image = encode_image(image, size_limit)
    return base64_image

def read_result_cache(cache_path = './cache/result_cache.json'):
    """Read the result cache from file"""
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        else:
            # Initialize directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            return []
    except Exception as e:
        logger.error(f"Error reading result cache: {e}")
        return []

def write_result_cache(cache_entry, cache_path = './cache/result_cache.json'):
    """Append a new entry to the result cache file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        # Read existing cache
        cache = read_result_cache()
        
        # Append new entry
        cache.append(cache_entry)
        
        # Write back to file
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.error(f"Error writing to result cache: {e}")

def clear_result_cache(cache_path = './cache/result_cache.json'):
    """Clear the result cache file"""
    # if the cache file is not exist, create it
    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump([], f)
    else:
        with open(cache_path, 'w') as f:
            json.dump([], f)

def functions_call(selected_tool, additional_requirements, tool_input):
    """
    Call the function with the given tool name, additional requirements, and tool input.
    """
    tool_function = FUNCTIONS[selected_tool](requirements_message=additional_requirements, **tool_input)
    return tool_function

def select_best_tool(  
    input_for_heuristic_search: dict,   
    dfs_sequence: list,   
    available_tools: list,   
    context: dict = None,  
    Temp_Workflow_meta_info: dict = None,  
):  
    """    
    
    Args:  
        input_for_heuristic_search (dict): Current search input  
        dfs_sequence (list): History of depth-first search  
        available_tools (list): Available workflows  
        context (dict): Additional context information  
    
    Returns:  
        tuple: (selected_tool, remaining_steps)  
    """  
    comprehensive_context = {  
        "current_input": input_for_heuristic_search,     
        "context": context["Used_Workflow"]  
    }  
    
    prompt_messages = [  
        {  
            "role": "system",  
            "content": f"""
            # Role
            You are an advanced Workflow Selection Agent. Your task is to carefully plan a sequential use of workflows based on the current context. This sequence represents the steps required to complete the task by executing workflows in order. You need to Think Step by Step.  
            
            # Instructions
            Strictly follow these guidelines:  
            1. Carefully analyze the current input and search history.  
            2. Evaluate the applicability of available workflows.  
            3. *Must* consider previous failed workflow attempts—workflows that have been recorded as incapable of completing a task should not be selected again.
            4. Apply logical reasoning: if a workflow has been recorded as failing for a follow-up task (not the current one due to possible workflow errors), selecting a similar workflow should be done with caution.  
            5. Achieving the desired result is the top priority! If there are no suitable workflows to proceed with the task, but reordering the workflow sequence can help, the generation order requested by the user can be adjusted when necessary.
            6. If no single workflow can advance the task, think deeply and creatively—combine multiple workflows and execute them sequentially to complete the task.
            7. Do not arbitrarily select workflows—only choose them if they can advance the task. If no suitable workflow can proceed with the task, return a failure signal.  
            8. Based on the planned workflow sequence, determine the remaining steps (remaining_steps). Think step by step: if only one workflow call is needed, remaining_steps = 0. If two sequential workflow calls are needed, remaining_steps = 1, and so on.
            9. You need to be aware of the difference between instructions and prompts. Instructions are descriptions of tasks, while prompts are just descriptions of the images in the tasks. For example, a task to remove an object should have the prompt "The Object" instead of "Remove the object"
            10. Analyze whether the user has additional requirements for the generated result, such as video duration, resolution, frame rate, upscaling factor, object placement, position, or any other requirements not covered by tool_input.
            11. Note: Additional requirements for generation results do not include vague terms like high quality, seamless integration, without visible artifacts, etc.
            12. If the context indicates that a certain workflow cannot complete the task, it is strictly forbidden to select that workflow again. Instead, try a more complex, multi-step workflow chain to solve the problem. For example, if a direct "Replace Object" workflow fails, do not select the same workflow again. Instead, consider: Using another workflow with object replacement functionality; Or First utilizing a masking workflow to generate a mask based on a prompt, then using an inpainting workflow with the masked image as input.
            13. Use past failed results to refine workflow selection and modify additional requirement parameters. For example, if an evaluation shows that the generated video duration is too short, recognize the user's intended duration and add it to additional_requirements.

            # Tools
            You have access to the following tools: {json.dumps(TOOLS)}

            ## Workflows and its Function and Input Parameters *IMPORTANT*
            You have access to the following Workflows:
            {Temp_Workflow_meta_info}
            ## ATTENTION
            - It is important to distinguish between instructions and prompts. Prompts only contain generated content, not requirements. For example, "Create a video of the cityscape with the perspective changing based on the image" is not allowed in the prompt generated by Image to Video. Correct prompt should be "cityscape". For example, inpaint the red apple to green apple is not allowed. Correct prompt should be "green apple".
            - Double check, and ensure the format of the tool_input follow the Tool and Workflow's JSON schema.
            - Parameters not mentioned need to be kept by default and added to the additional requirements. For example, there is no mention of changing the resolution(Upscale means the resolution is increased) and video duration (even if the frame rate is increased), but they need to be kept and added to the additional requirements.
            - additional_requirements should only contain the additional requirements used in the current step.
            - If a task is given in two steps but there is no corresponding workflow for one of the steps, the tasks can be integrated and completed in a single step. For example, if the user requests generating a video first and then upscaling it, but there is no workflow for video upscaling in the workflow library, the video can be generated at twice the resolution in one step to achieve the same effect.
            
            # Output
            ## Output Parameters
            After analyzing the workflow sequence, you must return a JSON object where:
            - tool and tool_input represent the first tool that needs to be invoked.
            - remaining_steps indicates how many more tool invocations are required to complete the task based on the planned sequence.
            - message provides a description of the current state and an explanation of the planned workflow.
            - additional_requirements represents any extra user requirements for the generated result at this step, such as video duration, resolution, frame rate, upscaling factor, object placement, position (e.g., "the building should be on the left"), or any other specifications not included in tool_input. Note: This does not include prompt-related attributes such as "detailed" or "high quality." The result should be returned as a string.

            ## Output Format 
            The return format should include the thought process wrapped in <think> </think>, followed by the JSON object wrapped in <json> </json>.
            {{
            "tool": ""Generate_Using_Workflow"",
            "tool_input": <parameters for the selected tool, Must match the tool's JSON schema>(*Do not* leave any placeholder),
            "remaining_steps": <number of steps remaining to complete the task>,
            "message": <The description of the current state and the explanation of the plan>,
            "additional_requirements": <additional requirements *In this step*, which may include resolution, frame rate, upsacle rate, etc. The format should be a string>
            }} 
            
            """
        },  
        {  
            "role": "user",   
            "content": json.dumps(comprehensive_context, ensure_ascii=False)  
        }  
    ]  
    
    try:    
        logger.info(f"Start Invoke Text")
        llm_response = invoke_text(prompt_messages)
        logger.info(f"llm_response Before Chain of Thought: {llm_response}")
        chain_of_thought = llm_response.split("<think>")[1].split("</think>")[0]
        json_content = json.loads(llm_response.split("<json>")[1].split("</json>")[0])

        if json_content:  
            function_name = json_content['tool']  
            function_args = json_content['tool_input']
            remaining_steps = json_content['remaining_steps']  
            message = json_content['message']
            additional_requirements = json_content['additional_requirements']
            
            if not function_name:  
                logger.info(f"No suitable tool found. message: {message}")
                return "No tool can complete the task" , None, None, None, None, chain_of_thought

            if function_name not in available_tools:  
                return f"Selected tool {function_name} is not in available tools" , None, None, None, None, chain_of_thought
            
            return function_name, function_args, remaining_steps, message, additional_requirements, chain_of_thought
        else:
            logger.info(f"No JSON content found in the response")
            return None, None, None, None, None, chain_of_thought

    except Exception as e:  
        logger.info(f"Tool selection failed: {e}")
        raise ValueError(f"Cannot select tool: {e}")
