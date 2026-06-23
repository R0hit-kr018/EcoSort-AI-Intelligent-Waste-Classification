"""
IBM Granite Helper for EcoSort Vision AI
"""

import os
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# Load environment variables
load_dotenv()

API_KEY = os.getenv("IBM_API_KEY")
PROJECT_ID = os.getenv("IBM_PROJECT_ID")
IBM_URL = os.getenv("IBM_URL")

MODEL_ID = "meta-llama/llama-3-3-70b-instruct"

PARAMETERS = {
    
    "max_new_tokens": 200,
    "temperature": 0.5
}

try:
    credentials = Credentials(
        api_key=API_KEY,
        url=IBM_URL
    )

    granite_model = ModelInference(
        model_id=MODEL_ID,
        credentials=credentials,
        project_id=PROJECT_ID,
        params=PARAMETERS
    )

except Exception as e:
    print(f"Granite Initialization Error: {e}")
    granite_model = None


def get_granite_response(
    waste_type: str,
    category: str,
    disposal: str,
    impact: str
) -> str:
    """
    Generate sustainability advice using IBM Granite
    """

    if granite_model is None:
        return "IBM Granite is not available."

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are an environmental sustainability expert. Provide concise, clear, and actionable sustainability advice. "
        f"Output ONLY direct paragraphs in plain text. Do NOT write any code, SQL queries, or programming syntax. "
        f"Do NOT repeat yourself.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Waste Item: {waste_type}\n"
        f"Category: {category}\n"
        f"Standard Disposal: {disposal}\n"
        f"Impact: {impact}\n\n"
        f"Based on the above waste details, write exactly three short points in plain text (keep it under 100 words total):\n"
        f"1. A sustainability recommendation for reuse or minimization.\n"
        f"2. A brief environmental awareness message.\n"
        f"3. Proper disposal/recycling guidance.\n\n"
        f"Response:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

    try:

        response = granite_model.generate_text(
            prompt=prompt
        )

        return response.strip()

    except Exception as e:

        return f"Granite Generation Error: {str(e)}"


# Test Run
if __name__ == "__main__":

    result = get_granite_response(
        waste_type="Plastic Bottle",
        category="Recyclable Waste",
        disposal="Place in a plastic recycling bin",
        impact="Reduces landfill waste and plastic pollution"
    )

    print(result)