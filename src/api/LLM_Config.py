import google.generativeai as genai
import re

# Configure the Generative AI API
genai.configure(api_key="AIzaSyDCoRQd6i4yiK0q2Ec2oTeIZ5KmPheDjn8")

def get_completion_from_messages(
    system_message: str,
    user_message: str,
    temperature: float = 0.3,
) -> str:
    try:
        # Combine system and user messages
        combined_message = f"{system_message}\n\nUser Query: {user_message}"
        print("=== INPUT ===")
        print(f"Combined Message:\n{combined_message}")
        print(f"Temperature: {temperature}")
        
        # Generate model
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(
            contents=combined_message,
            generation_config={"temperature": temperature}
        )

        # Log raw response
        print("\n=== RAW OUTPUT ===")
        print(f"Response Object: {response}")

        # Ensure text is a string and clean it
        text = response.text if isinstance(response.text, str) else str(response.text)
        clean_text = re.sub(r'```json\n|\n```', '', text)
        
        # Log cleaned text
        print("\n=== CLEANED OUTPUT ===")
        print(f"Cleaned Text:\n{clean_text}")
        
        return clean_text

    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        print("\n=== ERROR ===")
        print(error_msg)
        return error_msg
