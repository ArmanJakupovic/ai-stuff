import openai
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Retrieve the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.error("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")
    exit(1)

# Initialize primary OpenAI client (for generating responses)
openai.api_key = api_key

def generate_response(query):
    """
    Uses the OpenAI ChatCompletion API to generate a response based on the input query.
    """
    logging.info("Generating response for the query.")
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ],
            max_tokens=8192,  # Adjust based on response length needs
            temperature=0.7
        )
        generated_response = response.choices[0].message.content.strip()
        logging.info("Generated response successfully.")
        return generated_response
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "Error in generating response."

def evaluate_response(query, response):
    """
    Uses the evaluator client to analyze the response for relevance, accuracy, clarity, and bias.
    Each criterion is evaluated based on prompts designed to provide feedback for refinement.
    """
    logging.info("Evaluating response on relevance, accuracy, clarity, and bias.")
    evaluation_prompts = {
        "relevance": f"Evaluate how well the following response addresses the question.\n"
                     f"Question: {query}\nResponse: {response}\nProvide a relevance score between 0 and 1.",
        "accuracy": f"Check the following response for factual accuracy based on general knowledge.\n"
                    f"Response: {response}\nProvide an accuracy score between 0 and 1.",
        "clarity": f"Evaluate the clarity of the following response. Is it understandable and free of ambiguity?\n"
                   f"Response: {response}\nProvide a clarity score between 0 and 1.",
        "bias": f"Check the following response for any detectable biases or assumptions.\n"
                f"Response: {response}\nProvide a bias score between 0 and 1 (lower is better for bias)."
    }
    
    metrics = {}
    for criterion, prompt in evaluation_prompts.items():
        logging.info(f"Evaluating {criterion}.")
        try:
            eval_response = openai.chat.completions.create(
                model="gpt-4o-mini",  # Adjust based on available model
                messages=[
                    {"role": "system", "content": "You are a response evaluator."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,  # Limit to short responses
                temperature=0.3  # Lower temperature for consistent evaluations
            )
            score_text = eval_response.choices[0].message.content.strip()
            # Extract numeric score from the evaluator's response
            score = float(score_text) if score_text.replace('.', '', 1).isdigit() else 0.5  # Default if parsing fails
            metrics[criterion] = round(score, 2)
            logging.info(f"{criterion.capitalize()} score: {metrics[criterion]}")
        except Exception as e:
            logging.error(f"Error evaluating {criterion}: {e}")
            metrics[criterion] = 0.5  # Default score in case of error
    
    return metrics

# Define thresholds for acceptable metrics
THRESHOLDS = {
    "relevance": 0.8,
    "accuracy": 0.9,
    "clarity": 0.8,
    "bias": 0.2
}

def reflection_routine(query, max_iterations=3):
    """
    Performs the self-reflection routine with iterative context.
    """
    logging.info("Starting reflection routine.")
    iteration = 0
    response = generate_response(query)
    
    while iteration < max_iterations:
        logging.info(f"\nIteration {iteration + 1}")
        logging.info(f"Generated Response: {response}")
        
        # Evaluate response based on actual evaluator feedback
        metrics = evaluate_response(query, response)
        logging.info(f"Evaluation Metrics: {metrics}")
        
        # Check if all metrics meet the thresholds
        if (
            metrics["relevance"] >= THRESHOLDS["relevance"] and
            metrics["accuracy"] >= THRESHOLDS["accuracy"] and
            metrics["clarity"] >= THRESHOLDS["clarity"] and
            metrics["bias"] <= THRESHOLDS["bias"]
        ):
            logging.info("Status: Final Approved - Response meets all evaluation criteria.")
            return response, "Final Approved"
        
        # Generate feedback prompt to refine response
        feedback_prompt = (
            f"Original Question: {query}\n"
            f"Previous Response: {response}\n"
            "Please review and improve the previous response based on the following criteria:\n"
            "- Relevance: Ensure the response fully addresses the question.\n"
            "- Clarity: Identify and resolve any ambiguities.\n"
            "- Accuracy: Ensure factual correctness.\n"
            "- Completeness: Add any necessary details or examples.\n"
            "- Bias: Eliminate unnecessary assumptions or biases.\n"
            "Provide a refined response that meets these standards."
        )
        logging.info("Requesting a refined response based on feedback criteria.")
        response = generate_response(feedback_prompt)
        iteration += 1

    if metrics["relevance"] >= THRESHOLDS["relevance"] - 0.1 and metrics["accuracy"] >= THRESHOLDS["accuracy"] - 0.1:
        logging.info("Status: Review Recommended - Close to criteria, but not all fully met.")
        return response, "Review Recommended"
    else:
        logging.info("Status: Unresolved")
        return response, "Unresolved"

if __name__ == "__main__":
    logging.info("Prompting user for input.")
    query = input("Ask a question: ")
    logging.info(f"User query: {query}")

    final_response, status = reflection_routine(query)
    logging.info(f"Final Response: {final_response}")
    logging.info(f"Status: {status}")
