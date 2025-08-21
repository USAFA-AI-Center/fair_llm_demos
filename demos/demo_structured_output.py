# demo_structured_output.py
import os
import asyncio
import json
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

"""
This module provides a tutorial on forcing an LLM to produce structured,
validated JSON output based on a Pydantic schema.

**The Use Case:**
LLMs excel at understanding natural language, but many software applications
require structured data (like JSON). This pattern is essential for any task
that involves "reading" text and "understanding" it well enough to fill out a
form. Examples include:
* Parsing a resume to extract skills and work history.
* Reading an email to create a calendar event.
* Processing a product review to extract a rating, features mentioned, and sentiment.

**The Workflow:**
1.  **Define a Schema:** We will define a `UserProfile` class using Pydantic.
    This class acts as our "form" or the desired structured format.
2.  **Engineer a Prompt:** We'll create a highly specific prompt that instructs
    the LLM to act as a data extraction engine. We will provide it with the
    unstructured text and the JSON schema of our Pydantic model.
3.  **Extract and Validate:** The agent will call the LLM, attempt to parse the
    resulting JSON string, and use the Pydantic model to validate it.
4.  **Retry on Failure:** If the LLM produces invalid JSON or data that doesn't
    fit the schema, the agent will automatically retry, feeding the error back
    to the LLM so it can correct its own mistake. This self-correction loop is
    vital for building robust systems.
"""

# --- Step 1: Import all necessary components from the FAIR-LLM framework ---
from fairlib import settings, Message, OpenAIAdapter
from fairlib.core.interfaces.llm import AbstractChatModel # Keep this as it's an interface, not a concrete class

from dotenv import load_dotenv
load_dotenv()

settings.api_keys.openai_api_key = os.getenv("OPENAI_API_KEY")
settings.api_keys.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# --- Step 2: Define the target data structure using Pydantic ---
# This class is our "schema." It defines the exact structure and data types
# we want the LLM to extract. Pydantic will automatically handle validation.

class UserProfile(BaseModel):
    name: str = Field(..., description="The full name of the user.")
    age: int = Field(..., description="The age of the user.")
    city: str = Field(..., description="The city where the user resides.")
    interests: List[str] = Field(..., description="A list of the user's interests or hobbies.")
    is_student: bool = Field(..., description="Whether the user is currently a student.")

# --- Step 3: Engineer a specialized prompt for JSON extraction ---
# This prompt is highly specific. It tells the LLM its role, what to do,
# and most importantly, the exact JSON schema it needs to follow.
EXTRACTION_PROMPT_TEMPLATE = """
You are a highly efficient information extraction assistant. Your sole responsibility
is to extract structured data from the provided unstructured text, and return a valid
JSON object that strictly conforms to the JSON schema below.

You must adhere to the following rules:
- Do NOT include any commentary, explanation, or extra text.
- Only return the raw JSON object—no Markdown, no preamble, no wrapping text.
- Ensure all required fields are present and conform to the correct data types.
- If a field is missing from the text or cannot be confidently extracted, use null or an appropriate default.
- Booleans must be true/false, not "yes"/"no".
- Dates must follow ISO 8601 format if specified in the schema (e.g., YYYY-MM-DD).
- Strings must not contain leading/trailing whitespace.

**Target JSON Schema:**
```json
{schema}
Unstructured Source Text:
{text}

Now extract the data and return ONLY the JSON object:
"""

# --- Step 4: Create a dedicated agent for data extraction ---
class ExtractionAgent:

    """
    An agent designed specifically for extracting structured data from text.
    It includes a retry mechanism to ensure the output is valid.
    """

    def __init__(self, llm: AbstractChatModel, max_retries: int = 3):
        """
        Initializes the ExtractionAgent.

        Args:
        llm: The language model to use for extraction.
        max_retries: The maximum number of times to retry on validation failure.
        """
        self.llm = llm
        self.max_retries = max_retries

    async def extract(self, text: str, output_model: BaseModel) -> Optional[BaseModel]:
        """
        Attempts to extract data from text, validate it against the output_model,
        and retries on failure.

        Args:
            text: The unstructured text to extract information from.
            output_model: The Pydantic model to validate against.

        Returns:
            An instance of the Pydantic model if successful, otherwise None.
        """
        # Get the JSON schema from the Pydantic model.
        schema = json.dumps(output_model.model_json_schema(), indent=2)
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(schema=schema, text=text)

        for attempt in range(self.max_retries):
            print(f"\n--- Extraction Attempt {attempt + 1}/{self.max_retries} ---")
            
            # Call the LLM with the specialized prompt.
            response_text = self.llm.chat([Message(role="system", content=prompt)])
            print(f"LLM Raw Output:\n{response_text}")

            # Attempt to parse the LLM's output into our Pydantic model.
            # This line will raise a ValidationError if the data is wrong.
            try:
                validated_output = output_model.model_validate_json(response_text)
                print("\n✅ Validation Successful!")
                return validated_output
                
            except (ValidationError, json.JSONDecodeError) as e:
                # If parsing or validation fails, we add the error to the prompt
                # and retry, helping the LLM to self-correct.
                print(f"❌ Validation Failed: {e}")
                prompt += (
                    f"\n\nPREVIOUS FAILED ATTEMPT'S OUTPUT:\n{response_text}"
                    f"\n\nERROR MESSAGE:\n{e}\n\nPlease correct the output and try again."
                )

        print("\nCould not extract valid data after multiple attempts.")
        return None

async def main():
    """The main function to set up and run the extraction demo."""
    
    # --- Step 5: Initialize the LLM from the framework's components ---
    print("Initializing components...")
    llm = OpenAIAdapter(
        api_key=settings.api_keys.openai_api_key,
        model_name=settings.models["openai_gpt4"].model_name
    )
    
    # --- Step 6: Create the agent and define the input text ---
    extraction_agent = ExtractionAgent(llm)

    unstructured_text = (
        "My name is Jane Doe, and I live in San Francisco. I'm 28 years old. I like puppies and girls!"
        "I'm also a student right now. In my free time, I enjoy painting, "
        "playing the guitar, and long-distance running. I'm a scientist of all things!  I like God - yeah, pretty sure I like God and going to church, talking with people (girls)... I guess."
    )
    print("--- Input Text ---")
    print(unstructured_text)

    # --- Step 7: Run the Extraction Process ---
    # We call the agent, telling it what text to parse and what Pydantic
    # model to use as the target schema.
    extracted_data = await extraction_agent.extract(
        text=unstructured_text,
        output_model=UserProfile
    )
    
    # --- Step 8: Display the Final, Structured Result ---
    if extracted_data:
        print("\n--- ✅ Final Structured Data ---")
        print(extracted_data.model_dump_json(indent=2))
        
        # You can now access the data like a normal Python object, fully typed
        # and validated.
        print(f"\nExtracted Name: {extracted_data.name}")
        print(f"Extracted Interests: {extracted_data.interests}")

if __name__ == "__main__":
    asyncio.run(main())
