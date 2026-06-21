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

    prompt = f"""
You are an environmental sustainability expert.

Waste Type:
{waste_type}

Waste Category:
{category}

Recommended Disposal:
{disposal}

Environmental Impact:
{impact}

Provide:

1. Sustainability Recommendation
2. Environmental Awareness Message
3. Proper Disposal Guidance

Keep the response under 120 words.
Use simple language.
"""

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