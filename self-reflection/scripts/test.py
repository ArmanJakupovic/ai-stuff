import openai
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Retrieve the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.error("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")
    exit(1)

openai.api_key = api_key

def generate_response(query):
    """
    Uses the OpenAI ChatCompletion API to generate a response based on the input query.
    """
    logging.info("Generating response for the query.")
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ],
            max_tokens=200,  # Adjust based on response length needs
            temperature=0.7
        )
        # generated_response = response.choices[0].message['content'].strip()
        generated_response = response.choices[0].message.content.strip()
        # generated_response = response
        logging.info("Response generated successfully.")
        return generated_response
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "Error in generating response."

if __name__ == "__main__":
    # Prompt user to enter a question
    logging.info("Prompting user for input.")
    query = input("Ask a question: ")
    logging.info(f"User query: {query}")

    response = generate_response(query)
    logging.info(f"Response: {response}")
    print("Response:", response)
