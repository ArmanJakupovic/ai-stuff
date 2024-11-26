from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import os
import logging
import json
import requests
from enum import Enum
from typing import TypedDict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EvaluationStatus(Enum):
    FINAL_APPROVED = "Final Approved"
    REVIEW_RECOMMENDED = "Review Recommended"
    MINIMAL_IMPROVEMENT = "Minimal Improvement"
    UNRESOLVED = "Unresolved"
    ERROR = "Error"

class EvaluationMetrics(TypedDict):
    relevance: float
    accuracy: float
    clarity: float
    bias: float
    logical_coherence: float

@dataclass
class ModelConfig:
    jan_api_url: str = "http://localhost:1337/v1/chat/completions"
    model_name: str = "trinity-v1.2-7b"
    eval_model_name: str = "trinity-v1.2-7b"
    max_tokens: int = 8192
    eval_max_tokens: int = 50
    temperature: float = 0.7
    evaluation_temperature: float = 0.3

@dataclass
class ThresholdConfig:
    relevance: float = 0.8
    accuracy: float = 0.9
    clarity: float = 0.8
    bias: float = 0.2
    logical_coherence: float = 0.9
    min_improvement: float = 0.05
    max_iterations: int = 3

class LLMSelfReflection:
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        threshold_config: Optional[ThresholdConfig] = None
    ):
        self.model_config = model_config or ModelConfig()
        self.threshold_config = threshold_config or ThresholdConfig()

    def generate_response(self, query: str) -> str:
        """
        Generates a response using the Ollama API.
        """
        # logger.info(f"Generating response for query: {query}")
        try:
            messages = [
                    {"role": "system", "content": """
                        You are a helpful assistant with strong logical reasoning skills. Your answers are factual, logical and concise.
                        For all responses:
                        1. State the given information clearly
                        2. Structure your reasoning step by step
                        3. Ensure each step follows logically from the previous ones
                        4. Consider all relevant relationships in the information
                        5. Verify your conclusion against the original information
                    """},
                    {"role": "user", "content": query}
                ]

            response = self._chat_completion(messages, self.model_config.model_name, False, self.model_config.max_tokens, self.model_config.temperature)

            generated_response = response['choices'][0]['message']['content'].strip()
            # logger.info(f"Generated response: {generated_response}")

            return generated_response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            raise

    def evaluate_response(self, query: str, response: str) -> EvaluationMetrics:
        """
        Evaluates the response based on multiple criteria.
        """
        logging.info(f"Evaluating response on relevance, accuracy, clarity, and bias.")
        evaluation_prompts = {
            "relevance": self._create_evaluation_prompt("relevance", query, response),
            "accuracy": self._create_evaluation_prompt("accuracy", query, response),
            "clarity": self._create_evaluation_prompt("clarity", query, response),
            "bias": self._create_evaluation_prompt("bias", query, response),
            "logical_coherence": self._create_evaluation_prompt("logical_coherence", query, response)
        }
        
        metrics: EvaluationMetrics = {}
        for criterion, prompt in evaluation_prompts.items():
            logging.info(f"Evaluating {criterion}.")
            try:
                score = self._get_evaluation_score(criterion, prompt)
                metrics[criterion] = round(score, 2)
                logger.info(f"{criterion.capitalize()} score: {metrics[criterion]}")
            except Exception as e:
                logger.error(f"Error evaluating {criterion}: {str(e)}", exc_info=True)
                metrics[criterion] = 0.5
        
        return metrics

    def _create_evaluation_prompt(self, criterion: str, query: str, response: str) -> str:
        """Creates a specialized prompt for each evaluation criterion."""
        prompts = {
            "relevance": f"""
                Evaluate the relevance of this response to the question.
                Question: {query}
                Response: {response}
                Consider:
                1. Does it directly address the main points of the question?
                2. Are there any tangential or irrelevant details?
                3. Is the scope appropriate?
                Provide a score between 0 and 1, where 1 is perfectly relevant.
            """,
            "accuracy": f"""
                Evaluate the factual accuracy of this response:
                {response}
                Consider:
                1. Are all stated facts verifiable?
                2. Are there any contradictions?
                3. Is the information up-to-date and correct?
                Provide a score between 0 and 1, where 1 is perfectly accurate.
            """,
            "clarity": f"""
                Evaluate the clarity of this response:
                {response}
                Consider:
                1. Is the language clear and unambiguous?
                2. Is the structure logical?
                3. Would a general audience understand it?
                Provide a score between 0 and 1, where 1 is perfectly clear.
            """,
            "bias": f"""
                Evaluate this response for bias:
                {response}
                Consider:
                1. Are there unsupported assumptions?
                2. Is the language neutral and objective?
                3. Are multiple perspectives considered where appropriate?
                Provide a score between 0 and 1, where 0 is completely unbiased.
            """,
            "logical_coherence": f"""
                Evaluate the logical reasoning in this response.
                Question: {query}
                Response: {response}
                
                Consider these universal logical principles:
                1. Are premises clearly stated?
                2. Is each logical step properly supported?
                3. Are all relationships between elements properly considered?
                4. Are conclusions properly derived from the premises?
                5. Is the reasoning complete and sound?
                
                Provide a score between 0 and 1, where 1 indicates perfect logical coherence.
            """
        }
        return prompts[criterion].strip()

    def _get_evaluation_score(self, criterion: str, prompt: str) -> float:
        """Gets an evaluation score for a specific criterion."""
        try:
            # Modify prompt to instruct the model to return a score between 0 and 1
            prompt_with_instruction = f"""
                You are a response evaluator. Please evaluate the following response based on the {criterion} criterion. 
                Return only a single floating-point score between 0 and 1 as a JSON object with the key 'score'.
                Example1:
                {{
                    "score": 0
                }}
                Example2:
                {{
                    "score": 0.45
                }}
                Example3:
                {{
                    "score": 1
                }}
                
                Response to evaluate:
                {prompt}
            """

            messages = [
                    {"role": "system", "content": "You are a response evaluator."},
                    {"role": "user", "content": prompt_with_instruction}
                ]
            
            # logger.info(f"messages: {messages}")

            eval_response = self._chat_completion(messages, self.model_config.model_name, False, self.model_config.eval_max_tokens, self.model_config.evaluation_temperature)

            logger.info(f"eval response: {eval_response['choices'][0]['message']['content'].replace('</s>', '').strip()}")

            score_data = json.loads(eval_response['choices'][0]['message']['content'].replace('</s>', '').strip())
            # logger.info(f"score raw data: {score_data}")
            return float(score_data.get("score", 0.5))
            
        except Exception as e:
            logger.error(f"Error getting evaluation score for {criterion}: {str(e)}", exc_info=True)
            return 0.5

    def reflection_routine(self, query: str) -> Tuple[str, EvaluationStatus]:
        """
        Performs the self-reflection routine with iterative improvements.
        """
        logger.info("Starting reflection routine")
        iteration = 0
        response = self.generate_response(query)
        previous_metrics = None
        
        while iteration < self.threshold_config.max_iterations:
            logger.info(f"Iteration {iteration + 1}")
            
            metrics = self.evaluate_response(query, response)
            
            if self._meets_thresholds(metrics):
                return response, EvaluationStatus.FINAL_APPROVED
            
            if previous_metrics and self._minimal_improvement(metrics, previous_metrics):
                return response, EvaluationStatus.MINIMAL_IMPROVEMENT
            
            response = self._generate_improved_response(query, response, metrics)
            previous_metrics = metrics
            iteration += 1
        
        return response, self._determine_final_status(metrics)

    def _meets_thresholds(self, metrics: EvaluationMetrics) -> bool:
        """Checks if all metrics meet their thresholds."""
        return (
            metrics["relevance"] >= self.threshold_config.relevance and
            metrics["accuracy"] >= self.threshold_config.accuracy and
            metrics["clarity"] >= self.threshold_config.clarity and
            metrics["bias"] <= self.threshold_config.bias
        )

    def _minimal_improvement(self, current: EvaluationMetrics, previous: EvaluationMetrics) -> bool:
        """Determines if the improvement between iterations is minimal."""
        return all(
            abs(current[metric] - previous[metric]) < self.threshold_config.min_improvement
            for metric in current.keys()
        )

    def _generate_improved_response(
        self,
        query: str,
        previous_response: str,
        metrics: EvaluationMetrics
    ) -> str:
        """Generates an improved response based on evaluation metrics."""
        improvement_prompt = self._create_improvement_prompt(query, previous_response, metrics)
        return self.generate_response(improvement_prompt)

    def _create_improvement_prompt(
        self,
        query: str,
        previous_response: str,
        metrics: EvaluationMetrics
    ) -> str:
        """Creates a detailed prompt for improving the response."""
        return f"""
            Original Question: {query}
            Previous Response: {previous_response}
            
            Current Metrics:
            - Relevance: {metrics['relevance']} (Target: {self.threshold_config.relevance})
            - Accuracy: {metrics['accuracy']} (Target: {self.threshold_config.accuracy})
            - Clarity: {metrics['clarity']} (Target: {self.threshold_config.clarity})
            - Bias: {metrics['bias']} (Target: {self.threshold_config.bias})
            - Logical Coherence: {metrics['logical_coherence']} (Target: {self.threshold_config.logical_coherence})

            For improving logical reasoning:
            1. State all given information clearly
            2. Show explicit reasoning steps
            3. Ensure each conclusion follows from previous steps
            4. Consider all relationships in the information
            5. Verify the final answer maintains logical consistency
            
            Please improve the response focusing on:
            {self._get_improvement_priorities(metrics)}
            
            Provide a refined response that addresses these aspects while maintaining the strengths of the original response.
        """.strip()

    def _get_improvement_priorities(self, metrics: EvaluationMetrics) -> str:
        """Determines which aspects need the most improvement."""
        priorities = []
        if metrics["logical_coherence"] < self.threshold_config.logical_coherence:
            priorities.append("- Strengthen the logical reasoning and step-by-step analysis")
        if metrics["relevance"] < self.threshold_config.relevance:
            priorities.append("- Better address the core question")
        if metrics["accuracy"] < self.threshold_config.accuracy:
            priorities.append("- Verify and correct factual statements")
        if metrics["clarity"] < self.threshold_config.clarity:
            priorities.append("- Improve clarity and structure")
        if metrics["bias"] > self.threshold_config.bias:
            priorities.append("- Reduce bias and assumptions")
        return "\n".join(priorities) or "- Maintain current quality while seeking minor improvements"

    def _determine_final_status(self, metrics: EvaluationMetrics) -> EvaluationStatus:
        """Determines the final status based on the metrics achieved."""
        if metrics["relevance"] >= self.threshold_config.relevance - 0.1 and \
           metrics["accuracy"] >= self.threshold_config.accuracy - 0.1:
            return EvaluationStatus.REVIEW_RECOMMENDED
        return EvaluationStatus.UNRESOLVED
    
    def _chat_completion(self, messages, model, stream=False, max_tokens=2048, temperature=0.7):
        url = self.model_config.jan_api_url

        headers = {
            "Content-Type": "application/json",
        }

        payload = {
            "messages": messages,
            "model": model,
            "stream": stream,
            "max_tokens": max_tokens,
            "stop": None,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "temperature": temperature,
            "top_p": 0.95
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        return response.json()


def main():
    try:
        reflector = LLMSelfReflection()
        query = input("Ask a question: ")
        logger.info(f"User query: {query}")
        
        response, status = reflector.reflection_routine(query)
        logger.info(f"Final Response: {response}")
        logger.info(f"Status: {status.value}")
        
    except Exception as e:
        logger.error(f"Error in main routine: {str(e)}", exc_info=True)
        return

if __name__ == "__main__":
    main()