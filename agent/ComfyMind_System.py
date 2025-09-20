import json
import os
import logging
import json
logger = logging.getLogger(__name__) 

from utils.model import invoke_text, invoke_vision
from utils.utils import select_best_tool, functions_call, load_image, read_result_cache, write_result_cache, clear_result_cache
from utils.eval_utils import load_video
from utils.tools import TOOLS
from utils.eval import check_generation_judgment
from script.evaluation_geneval import evaluate_geneval_image
import string

system_prompt_for_prompt_optimization = """
# Objective:
Determine if prompt optimization is needed:
If the task Do not have reference image or video, the prompt must be optimized(e.g. Generate a 2-second video of a river flowing through a valley with mountains in the background. The result should be a high-quality video.). If the task has reference image or video, do not optimize the prompt.
# Prompt Optimization Guidelines

## For T2I (Text-to-Image):A well-optimized prompt follows this structure:Prompt = Subject + Scene + Style + Camera Language + Atmosphere + Detail Enhancement
Subject: Clearly define the main subject, including characteristics, appearance, and actions. Example: "A charming 10-year-old Chinese girl, wearing a bright red dress, smiling under the sunlight."
Scene: Describe the environment, background elements, and setting. Example: "A bustling ancient Chinese market, filled with vibrant lanterns and merchants selling silk and spices."
Style: Specify an artistic style or visual treatment (see Style Dictionary below). Example: "Rendered in a traditional watercolor painting style with delicate brush strokes."
Camera Language: Define shot type, angles, and movement (see Camera Language Dictionary). Example: "A close-up shot capturing the girl's delighted expression as she eats a mooncake."
Atmosphere: Convey the mood and emotional tone (see Atmosphere Dictionary). Example: "Warm and nostalgic, evoking a sense of childhood happiness."
Detail Enhancement: Add refined details to enrich the composition. Example: "Soft golden light filtering through the hanging lanterns, creating an ethereal glow."

## For T2V / I2V (Text-to-Video, Image-to-Video):A well-optimized prompt follows this structure:Prompt = Subject + Scene + Motion + Camera Language + Atmosphere + Style
Subject: Describe the main character or object with specific attributes.Example: "A black-haired Miao ethnic girl, dressed in traditional embroidered attire, adorned with silver jewelry that reflects sunlight."
Scene: Define the background, setting, and environmental elements.Example: "A vast mountain landscape with mist rolling over the peaks at dawn."
Motion: Describe movement speed, style, and effect.Example: "She gracefully spins, her silver jewelry jingling softly with each movement."
Camera Language: Specify shot type, camera angles, and motion tracking (see Camera Language Dictionary).Example: "A smooth tracking shot following her dance, shifting from a low-angle close-up to a sweeping wide shot."
Atmosphere: Define the mood and ambiance (see Atmosphere Dictionary).Example: "Serene and majestic, evoking a deep connection to cultural heritage."
Style: Choose a distinct visual or artistic style (see Style Dictionary).Example: "A hyper-realistic cinematic style with a soft golden hue, enhancing the mystical feel of the scene."

## Final Prompt
Finally, combine all the elements(Subject, Scene, Motion, Camera Language, Atmosphere, Style) into one paragraph.

## Prompt Dictionary
1. Camera Language
Shot Types (Framing):
Close-up Shot: Captures fine details, expressions, or objects in high focus.Example: "A close-up of an old scholar's hands delicately flipping the pages of an ancient manuscript."
Medium Shot: Shows the subject from the waist up, providing more context.Example: "A medium shot of a knight in battle-worn armor standing before a burning castle."
Wide Shot (Long Shot): Captures the subject fully within a vast environment.Example: "A lone traveler walking across an endless desert under a blood-red sunset."
Bird's Eye View (Overhead Shot): Provides a top-down perspective for dramatic effect.Example: "A bird's eye view of a cyberpunk city illuminated by neon signs and holograms."
Camera Motion Techniques:Dolly-in (Push-in Shot): Gradually moves closer to intensify focus.Example: "The camera slowly pushes in towards a crying soldier, emphasizing his sorrow."
Pull-out (Zoom-out Shot): Moves backward to reveal a larger scene.Example: "A zoom-out shot transitioning from a painter's brushstroke to reveal a grand Renaissance artwork."
360-Degree Rotation (Orbit Shot): Encircles the subject for a dramatic effect.Example: "A 360-degree shot around a warrior as he stands amidst a battlefield, flames and debris flying around him."
Tracking Shot (Follow Shot): Follows a subject in motion dynamically.Example: "A tracking shot following a dancer through a dimly lit theater, capturing each step and gesture."

2. Atmosphere (Mood & Emotion)
Energetic / Joyful / Uplifting: Bright lighting, vibrant colors, and lively movement.Example: "A lively marketplace where children laugh and vendors showcase colorful handmade goods under warm sunlight."
Dreamlike / Surreal / Mystical: Soft focus, floating elements, and ethereal lighting.Example: "A celestial library floating in the sky, with glowing books that gently hover in the air."
Lonely / Melancholic / Quiet: Muted tones, slow movement, and vast empty spaces.Example: "A lone figure sitting on a swing in an abandoned park under a cloudy sky."
Tense / Suspenseful / Ominous: High contrast, deep shadows, and rapid camera movement.Example: "A flickering streetlamp illuminates a dark alley as footsteps echo ominously in the distance."
Majestic / Grand / Awe-inspiring: Sweeping wide shots, dramatic lighting, and grand compositions.Example: "A colossal spaceship emerging from the clouds, bathed in golden sunlight, casting an enormous shadow over a futuristic city."

3. Style (Artistic Direction)
Cyberpunk: Neon lights, dark cityscapes, high-tech elements.Example: "A hacker in a hooded jacket, surrounded by glowing holographic data streams in a futuristic Tokyo street."
Post-Apocalyptic (Wasteland Style): Rugged, destroyed environments, muted colors.Example: "A lone wanderer in tattered clothes walks through a desolate wasteland, carrying a rusted rifle."
Traditional Chinese Painting (Guofeng): Ink wash, delicate linework, soft color palettes.Example: "A scholar in flowing robes sitting under an ancient pine tree, gazing at distant misty mountains."
Felt Animation Style: Soft, handmade textures, childlike charm.Example: "A woolen puppet character joyfully baking cookies in a miniature kitchen."
Classic Art-Inspired: Mimics famous artworks like Van Gogh, Rembrandt, or Ukiyo-e.Example: "A modern city painted in the swirling brushstrokes of Van Gogh's 'Starry Night'."

# Output
If optimization is not required, only None is output. If optimization is required, the output only includes the optimized prompt, which is wrapped by <optimized_prompt> </optimized_prompt>
"""


system_prompt_for_instruction_analysis = """
You are a professional Requirements Analyst. Your primary responsibility is to structure and analyze user-generated requirements, ensuring that key details, constraints, and expectations are accurately captured.
Your output should focus on highlighting considerations for subsequent planning rather than providing planning recommendations. Specifically, your analysis should:
Identify key constraints and specifications (e.g., resolution, frame rate, duration, dependencies).
Ensure alignment with given references or guidelines rather than assuming defaults.
Highlight potential ambiguities or missing details that require clarification.
Output:
1. Wrap your output in <analysis> </analysis>
2. *One* sentence about Analysis of the user's instruction. Accurate, Concise.
"""


system_prompt_for_updata_input = """
You are a helpful assistant that can update the input of the user. I will give you the current input and the output of the tool. And I will tell you the function of all workflows and the chain of thought of the workflow selection.
Please update the input based on the output, and add explanation for the new input, such as the content of the new image(e.g. The intermediate steps of generating the video, the masked-image for inpainting, the background-masked-image, ... Etc.).
ATTENTION:
1. Your output should be a JSON object. Totally follow the JSON schema of the input.
2. You should read the information of the workflow and the chain of thought, accroding the workflow's function guess what steps you have completed and what steps you may next complete. Then add the content of the new added input parameters and them to instruction. This may include the content of the new prompt/image/video in output. 
3. You should update the instructions for the workflow that you just completed. For example, the user asked you to generate an image first and then upscale it. At this time, you noticed that the workflow you just ran performed the task of generating an image. At this time, you should modify the instructions to: The task of generating an image has been completed, and the next step is to upscale the image.
4. You should pay attention to the timeliness of the user's instructions. For example, The instruction:generate an image with a resolution of XXX or a video with a duration of XXX. Such instructions are permanent. Therefore, it should continue to be passed, and at the same time remind the subsequent workflow to continue to maintain the generated resolution and video time.
5. *IMPORTANT* You MUST add information and introduction for the new generated file to "file_meta_info"(*Do not* leave it empty).
6. *IMPORTANT* You MUST maintain ALL Previous step 'file_meta_info' of the previous files(e.g. the image, the video, ...etc.). If the previous files are not mentioned in the 'file_meta_info', you should add them to the 'file_meta_info'. E.g: This image is the original input image for removing the background.
7. *IMPORTANT* You must measure the gap between the current generated content and the user's target generated content to provide information for subsequent tool calls. For example, the user requires the generation of a green banana, but the banana currently generated is yellow. The color may need to be modified later.
Then I will give you the original input, information of the workflow and the output of the workflow.
"""


class ComfyMind(object):
    def __init__(self, eval_agent=None, meta_info_site=None, preprocessing=None):

        self.eval_agent = eval_agent
        self.workflow_meta_info_path = meta_info_site
        self.preprocessing = preprocessing

        self.dist_for_node_label = {}  
        self.used_node_labels = set()  
        self.counter_for_node_label = {} 
        self.letters = list(string.ascii_uppercase) 
        self.repeat_times = 1
        self.MAX_LENGTH = 20
        self.MAX_RESULT_CACHE_SIZE = 5
        self.result_cache_path = './cache/result_cache.json'
        self.get_input_description = True
        self.failure_result = None

        with open(self.workflow_meta_info_path, 'r', encoding='utf-8') as f:
            self.Workflow_meta_info = json.load(f)

    def __call__(self, task: dict) -> str:
        clear_result_cache(self.result_cache_path)
        images_path = []
        videos_path = []
        file_meta_info = {}
        for i in range(self.repeat_times):
            optimized_prompt = None
            analysis_of_instruction = None
            instruction = task['instruction']
            generation = None

            if self.get_input_description and (task['resource1'] or task['resource2']):
                for resource in [task['resource1'], task['resource2']]:
                    if resource and (resource.endswith('.png') or resource.endswith('.jpg') or resource.endswith('.jpeg')):
                        image_base64 = load_image(resource)
                        content = [
                            {
                                "type": "text",
                                "text": f"""Given the input image, please describe the content of the image in detail. Wrap your output in <input_description> </input_description>
                                """
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                        prompt_messages = [
                            {
                                "role": "user",
                                "content": content
                            }
                        ]
                        input_description = invoke_vision(prompt_messages)
                        input_description = input_description.split('<input_description>')[1].split('</input_description>')[0] if '<input_description>' in input_description else None
                        logger.info(f"Input Image Description: {input_description}")
                        images_path.append(resource)
                        file_meta_info[resource] = f'Given Reference Image Input: {input_description}'
                    elif resource and resource.endswith('.mp4'):
                        base64_frames, meta_info = load_video(resource) # The first {reference_frame_count} images are the frames sampled from the input reference
                        content = [{
                            'type': 'text',
                            'text': 'Given the input video, The images are the frames sampled from the input reference video. Please describe the content of the video in detail. Wrap your output in <input_description> </input_description>'
                        }]
                        for base64_frame in base64_frames:
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_frame}"
                                }
                            })
                        message = [{
                            "role": "user",
                            "content": content
                        }]
                        input_description = invoke_vision(message)
                        input_description = input_description.split('<input_description>')[1].split('</input_description>')[0] if '<input_description>' in input_description else None
                        logger.info(f"Input Video Description: {input_description}")
                        videos_path.append(resource)
                        file_meta_info[resource] = f'Given Reference Video Input: {input_description}'

            if self.preprocessing == 'prompt_optimization':
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt_for_prompt_optimization
                    },
                    {
                        "role": "user",
                        "content": instruction
                    }
                ]
                logger.info(' PreProcessing : Prompt Optimization '.center(80, '-'))
                generation = invoke_text(messages)
                # If optimization is not required, only None is output. If optimization is required, the output only includes the optimized prompt, which is wrapped by <optimized_prompt> </optimized_prompt>
                optimized_prompt = generation.split('<optimized_prompt>')[1].split('</optimized_prompt>')[0] if '<optimized_prompt>' in generation else None
            
            elif self.preprocessing == 'instruction_analysis':
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt_for_instruction_analysis
                    },
                    {
                        "role": "user",
                        "content": instruction
                    }
                ]
                generation = invoke_text(messages)
                # Wrap output in <analysis> </analysis>
                analysis_of_instruction = generation.split('<analysis>')[1].split('</analysis>')[0] if '<analysis>' in generation else None
                logger.info(f" PreProcessing : Instruction Analysis ".center(80, '-'))

            input_for_heuristic_search = {
                "prompt": optimized_prompt if optimized_prompt else instruction,
                "instruction": analysis_of_instruction if analysis_of_instruction else instruction,
                "images": images_path,
                "videos": videos_path,
                "file_meta_info": file_meta_info
            }

            dfs_sequence = []
            available_tools = [tool["name"] for tool in TOOLS]

            logger.info("Start Search ...")

            result = self._Heuristic_Search(
                input_for_heuristic_search, 
                dfs_sequence, 
                available_tools, 
                task=task,
                node_id=None,
                graph=[],
            )
            
            return result


    def assign_label(self):
        for letter in self.letters:
            if letter not in self.used_node_labels:
                self.dist_for_node_label[letter] = letter
                self.used_node_labels.add(letter)
                return letter

        for letter in self.letters:
            num = self.counter_for_node_label.get(letter, 1)
            new_label = f"{letter}_{num}"
            if new_label not in self.used_node_labels:
                self.dist_for_node_label[new_label] = new_label
                self.used_node_labels.add(new_label)
                self.counter_for_node_label[letter] = num + 1
                return new_label


    def extract_json(self, s):
        stack = 0
        start = s.find('{')
        if start == -1:
            return None

        for i in range(start, len(s)):
            if s[i] == '{':
                stack += 1
            elif s[i] == '}':
                stack -= 1
                if stack == 0:
                    return s[start:i + 1]
        return None


    def _Heuristic_Search(  
        self,
        input_for_heuristic_search: dict,   
        dfs_sequence: list,
        available_tools: list,
        task: dict,
        node_id = None,
        graph = None,
    ):  
    
        # Initialize the node list
        if dfs_sequence == []:
            # Refresh the Node
            self.dist_for_node_label = {}
            self.used_node_labels = set()
            self.counter_for_node_label = {}
            logger.info("Refreshed the Node List".center(80, '-'))

        if node_id == None:
            node_id = self.assign_label()

        result_cache = read_result_cache(self.result_cache_path)
        
        # Track false judgments
        false_judgment_count = 0
        Temp_Workflow_meta_info = self.Workflow_meta_info.copy()

        if isinstance(input_for_heuristic_search, str):
            try:
                input_for_heuristic_search = json.loads(input_for_heuristic_search)
            except json.JSONDecodeError:
                input_for_heuristic_search = {
                    "prompt": input_for_heuristic_search,
                    "instruction": input_for_heuristic_search,
                    "images": [],
                    "videos": [],
                    "file_meta_info": {}
                }

        current_context = {  
            "initial_input": input_for_heuristic_search.copy(),  
            "Used_Workflow": [],
            "Generation_Evaluations": []
        }  

        if len(result_cache) >= self.MAX_RESULT_CACHE_SIZE:
            return {  
                "status": "failed",   
                "output": None,
                "error_message": "Maximum result cache reached",  
                "dfs_sequence": dfs_sequence,
                "graph": graph,
                "judgment": 'False',
                "evaluation": 'Maximum result cache reached',
            }


        while True:  

            # Length limitation 
            if len(dfs_sequence) >= self.MAX_LENGTH:  
                return {  
                    "status": "Completed",   
                    "output": None,   
                    "error_message": "Maximum search length reached",  
                    "dfs_sequence": dfs_sequence,
                    "graph": graph,
                    "judgment": 'False',
                    "evaluation": 'Maximum search length reached',
                }    

            # Step 1: Analyze the current state  
            logger.info("Analyzing current state".center(80, '-'))  
            logger.info(f"Current Input: {input_for_heuristic_search}")  
            logger.info(f"Current Context: {current_context}")

            # Step 2: Planning Agent
            try:  
                selected_tool, tool_input, Remaining_steps, message, additional_requirements, chain_of_thought = select_best_tool(  
                    input_for_heuristic_search,   
                    dfs_sequence,   
                    available_tools,   
                    context=current_context,  
                    Temp_Workflow_meta_info=Temp_Workflow_meta_info  
                )

                if selected_tool is None or tool_input is None or tool_input == {}:
                    return {
                        "status": "failed",
                        "output": None,
                        "error_message": "No workflow can complete the task",
                        "dfs_sequence": dfs_sequence,
                        "graph": graph,
                        "judgment": 'False',
                        "evaluation": 'No workflow can complete the task',
                    }

                logger.info("Workflow Select End".center(80, '-'))
                logger.info(f"Selected Workflow: {selected_tool}")  
                logger.info(f"Workflow Input: {tool_input}, {message}")

            except Exception as e:  
                logger.info(f"Workflow selection error: {str(e)}")
                return {  
                    "status": "failed",   
                    "output": None,   
                    "error_message": f"Workflow selection error: {str(e)}",  
                    "dfs_sequence": dfs_sequence,
                    "graph": graph,
                    "judgment": 'False',
                    "evaluation": 'Workflow selection error',
                }  


            # Step 3: Execution Agent
            tool_output = None 
            try:
                print("Workflow execution process:")
                print(f"Selected tool: {selected_tool}")
                print(f"Additional requirements: {additional_requirements}")
                print(f"Tool input: {tool_input}")
                tool_output = functions_call(selected_tool, additional_requirements, tool_input)
                logger.info(f"Workflow Output: {tool_output}")
            except Exception as e:  
                tool_message = str(e)  
                logger.info(f"Workflow execution error: {tool_message}")
                tool_output = {}

            if str(tool_output) == '{}' or tool_output is None:  
                workflow_name = tool_input.get('workflow_name', 'Unknown workflow')
                logger.info(f"Workflow {workflow_name} failed".center(80, '-'))

                logger.info(f"Refreshing Temp_Workflow_meta_info: {Temp_Workflow_meta_info}")
                Temp_Workflow_meta_info[tool_input["workflow_name"]] = 'Used, Failed'

                # Record tool attempt errors, but do not return directly    
                dfs_sequence.append({"tool": selected_tool, "status": "failed"})  
                current_context["Used_Workflow"].append({"Workflow_Name": tool_input["workflow_name"], "message": 'Error: No output was obtained. There is a problem with the workflow under the current output.'})  
                continue  

            next_node_id = self.assign_label()
            graph.append((f'{tool_input["workflow_name"]}', node_id, next_node_id))

            # Record successful tool calls  
            dfs_sequence.append({"tool": selected_tool, "status": "success", "input": tool_input})  
            Temp_Workflow_meta_info[tool_input["workflow_name"]] = 'Used'
            logger.info(f"Refreshing Temp_Workflow_meta_info: {Temp_Workflow_meta_info}")
            
            # Update the workspace
            logger.info(f"Update the workspace".center(80, '-'))
            workflow_info = self.Workflow_meta_info[tool_input["workflow_name"]]

            try:
                image_base64 = load_image(tool_output['outputs'][0]['images'])
            except Exception as e:
                image_base64 = None

            if image_base64 is None:
                prompt_messages = [
                    {
                        "role": "system",
                        "content": f"{system_prompt_for_updata_input}"
                    },
                    {
                        "role": "user",
                        "content": f"""Given the used workflow and its information:{workflow_info}; 
                        Given the original input: {input_for_heuristic_search}.
                        Information of the workflow: {tool_input}.
                        The output of the workflow: {tool_output}.
                        Please update the input.
                        """                
                    }
                ]
                temp_input_for_heuristic_search = invoke_text(prompt_messages)
            else:
                content = [
                    {
                        "type": "text",
                        "text": f"""Given the used workflow and its information:{workflow_info}; 
                        Given the original input: {input_for_heuristic_search}.
                        Information of the workflow: {tool_input}.
                        The output of the workflow: {tool_output}.
                        Please update the input.
                        """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]

                prompt_messages = [
                    {
                        "role": "system",
                        "content": f"""{system_prompt_for_updata_input}; 
                        IMPORTANT: The 'file_meta_info' should be described in detail based on the given image and the functionality of the used workflow, including its content, style, and most importantly, its relationship with other provided images (e.g., same content with a different style or the same style but a newly generated image with different content).
                        """
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ]
                temp_input_for_heuristic_search = invoke_vision(prompt_messages)

            logger.info(f"temp_input_for_heuristic_search: {temp_input_for_heuristic_search}")

            try:
                temp_input_for_heuristic_search = self.extract_json(temp_input_for_heuristic_search)
            except Exception as e:
                logger.info(f"Error extracting JSON: {e}")
                temp_input_for_heuristic_search = None



            # Step 4: Evaluation Agent : Check if the task is completed  
            if Remaining_steps == 0:  
                logger.info(f"{'Task completed successfully, and waiting for verification ...'.center(80, '-')}")
                try:
                    if self.eval_agent == 'normal':
                        print("Evaluation agent process: normal")
                        evaluation, judgment, result_path = check_generation_judgment(tool_output, task)
                    elif self.eval_agent == 'geneval':
                        print("Evaluation agent process: geneval")
                        result_json = evaluate_geneval_image(image_path=tool_output['outputs'][0]['images'], metadata=task)
                        judgment = str(result_json['correct'])
                        result_path = result_json['filename']
                        evaluation = f'{result_json["correct"]}:{result_json["reason"]}'
                        
                    else:
                        # Skip evaluation
                        evaluation, judgment, result_path = check_generation_judgment(tool_output, task, skip_judgment=True)
                    
                    print(f"Evaluation: {evaluation}")
                    print(f"Judgment: {judgment}")
                    print(f"Result path: {result_path}")
                    
                    if judgment == 'True':
                        cache_entry = {
                            "workflow_name": tool_input["workflow_name"],
                            "result": tool_output,
                            "judgment": judgment,
                            "evaluation": evaluation
                        }
                        
                        result_cache.append(cache_entry)
                        write_result_cache(cache_entry, self.result_cache_path)

                        return {  
                            "status": "completed",   
                            "output": result_path,
                            "error_message": None,  
                            "dfs_sequence": dfs_sequence,
                            "graph": graph,
                            "judgment": judgment,
                            "evaluation": evaluation,
                        }
                    else:
                        false_judgment_count += 1
                        cache_entry = {
                            "workflow_name": tool_input["workflow_name"],
                            "result": tool_output,
                            "judgment": judgment,
                            "evaluation": evaluation
                        }
                        
                        result_cache.append(cache_entry)
                        write_result_cache(cache_entry, self.result_cache_path)
                        
                        current_context["Used_Workflow"].append({"Workflow_Name": tool_input["workflow_name"], "message": f"Evaluation: {evaluation}"})
                        
                        if false_judgment_count >= self.MAX_FALSE_JUDGMENT_COUNT:
                            return {
                                "status": "failed",
                                "output": result_path,
                                "error_message": "Accumulated too many failed generation attempts",
                                "dfs_sequence": dfs_sequence,
                                "graph": graph,
                                "judgment": judgment,
                                "evaluation": evaluation,
                            }
                        
                        continue

                except Exception as e:
                    logger.info(f"Evaluation Error: {e}")
                    current_context["Used_Workflow"].append({  
                        "Workflow_Name": tool_input["workflow_name"],  
                        "message": "Does not Generate the corresponding format result"
                    })   
                    continue    

            elif Remaining_steps > 0:  
                logger.info("Generation successful but task not completed.")  
                # Continue with recursive call
                result = self._Heuristic_Search(  
                    temp_input_for_heuristic_search,   
                    dfs_sequence,   
                    available_tools,
                    task=task,
                    node_id=next_node_id,
                    graph=graph,
                )  
                
                if result["status"] == "completed":  
                    logger.info(f"DFS Sequence: {result['dfs_sequence']}")
                    return result  
                else:   
                    current_context["Used_Workflow"].append({  
                        "Workflow": tool_input,  
                        "message": "Cannot complete subsequent tasks"
                    })  
                    
                    continue  
        