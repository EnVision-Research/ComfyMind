import openai
import base64
import re
from typing import Dict, Any
from utils.model import invoke_vision


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_scores(evaluation_text: str) -> Dict[str, float]:
    score_pattern = r"\*{0,2}(Consistency|Realism|Aesthetic Quality)\*{0,2}\s*[:：]?\s*(\d)"
    matches = re.findall(score_pattern, evaluation_text, re.IGNORECASE)

    scores = {
        "consistency": 9.9,
        "realism": 9.9,
        "aesthetic_quality": 9.9
    }

    for key, value in matches:
        key = key.lower().replace(" ", "_")
        if key in scores:
            scores[key] = float(value)

    return scores


def build_evaluation_messages(prompt_data: Dict[str, Any], image_base64: str) -> list:
    return [

        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a professional Vincennes image quality audit expert, please evaluate the image quality strictly according to the protocol."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Please evaluate strictly and return ONLY the three scores as requested.

# Text-to-Image Quality Evaluation Protocol

## System Instruction
You are an AI quality auditor for text-to-image generation. Apply these rules with ABSOLUTE RUTHLESSNESS. Only images meeting the HIGHEST standards should receive top scores.

**Input Parameters**  
- PROMPT: [User's original prompt to]  
- EXPLANATION: [Further explanation of the original prompt] 
---

## Scoring Criteria

**Consistency (0-2):**  How accurately and completely the image reflects the PROMPT.
* **0 (Rejected):**  Fails to capture key elements of the prompt, or contradicts the prompt.
* **1 (Conditional):** Partially captures the prompt. Some elements are present, but not all, or not accurately.  Noticeable deviations from the prompt's intent.
* **2 (Exemplary):**  Perfectly and completely aligns with the PROMPT.  Every single element and nuance of the prompt is flawlessly represented in the image. The image is an ideal, unambiguous visual realization of the given prompt.

**Realism (0-2):**  How realistically the image is rendered.
* **0 (Rejected):**  Physically implausible and clearly artificial. Breaks fundamental laws of physics or visual realism.
* **1 (Conditional):** Contains minor inconsistencies or unrealistic elements.  While somewhat believable, noticeable flaws detract from realism.
* **2 (Exemplary):**  Achieves photorealistic quality, indistinguishable from a real photograph.  Flawless adherence to physical laws, accurate material representation, and coherent spatial relationships. No visual cues betraying AI generation.

**Aesthetic Quality (0-2):**  The overall artistic appeal and visual quality of the image.
* **0 (Rejected):**  Poor aesthetic composition, visually unappealing, and lacks artistic merit.
* **1 (Conditional):**  Demonstrates basic visual appeal, acceptable composition, and color harmony, but lacks distinction or artistic flair.
* **2 (Exemplary):**  Possesses exceptional aesthetic quality, comparable to a masterpiece.  Strikingly beautiful, with perfect composition, a harmonious color palette, and a captivating artistic style. Demonstrates a high degree of artistic vision and execution.

---

## Output Format

**Do not include any other text, explanations, or labels.** You must return only three lines of text, each containing a metric and the corresponding score, for example:

**Example Output:**
Consistency: 2
Realism: 1
Aesthetic Quality: 0

---

**IMPORTANT Enforcement:**

Be EXTREMELY strict in your evaluation. A score of '2' should be exceedingly rare and reserved only for images that truly excel and meet the highest possible standards in each metric. If there is any doubt, downgrade the score.

For **Consistency**, a score of '2' requires complete and flawless adherence to every aspect of the prompt, leaving no room for misinterpretation or omission.

For **Realism**, a score of '2' means the image is virtually indistinguishable from a real photograph in terms of detail, lighting, physics, and material properties.

For **Aesthetic Quality**, a score of '2' demands exceptional artistic merit, not just pleasant visuals.

--- 
Here are the Prompt and EXPLANATION for this evaluation:
PROMPT: "{prompt_data['Prompt']}"
EXPLANATION: "{prompt_data['Explanation']}"
Please strictly adhere to the scoring criteria and follow the template format when providing your results."""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        }
    ]


from openai import OpenAI
def evaluate_single_image_wise(
    image_path: str,
    prompt_data: Dict[str, Any]
) -> Dict[str, Any]:
    try:
        base64_image = encode_image(image_path)
        messages = build_evaluation_messages(prompt_data, base64_image)

        import yaml
        import os
        with open('./config.yaml', 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            VISION_BASE_URL = config['vision']['base_url']
            VISION_API_KEY = config['vision']['api_key']
            VISION_MODEL_NAME = config['vision']['model_name']
            VISION_HYPER_PARAMETER = config['vision']['hyper_parameter']


        client = OpenAI(
            base_url=VISION_BASE_URL,
            api_key=VISION_API_KEY
        )


        response = client.chat.completions.create(
            model=VISION_MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=2000
        )
        evaluation_text = response.choices[0].message.content


        scores = extract_scores(evaluation_text)
        normalized_mean = round(
            (scores["consistency"]*0.7 + scores["realism"]*0.2 + scores["aesthetic_quality"]*0.1) / 2, 3)

        return {
            "evaluation_text": evaluation_text,
            "consistency": scores["consistency"],
            "realism": scores["realism"],
            "aesthetic_quality": scores["aesthetic_quality"],
            "normalized_mean": normalized_mean
        }

    except Exception as e:
        return {
            "evaluation_text": f"Evaluation failed: {str(e)}",
            "consistency": 9.9,
            "realism": 9.9,
            "aesthetic_quality": 9.9,
            "normalized_mean": 9.9
        }
