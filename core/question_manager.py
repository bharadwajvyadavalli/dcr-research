import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Suppress verbose question manager logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class QuestionManager:
    """Manages questions and response caching per provider"""

    def __init__(self, questions_file: str = "questions.json", cache_dir: str = "cache"):
        self.questions_file = questions_file
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def save_questions(self, questions: List[str]) -> None:
        """
        Save questions to a JSON file before making API calls

        Args:
            questions: List of question strings
        """
        data = {
            'timestamp': datetime.now().isoformat(),
            'total_questions': len(questions),
            'questions': questions
        }

        with open(self.questions_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(questions)} questions to {self.questions_file}")

    def load_questions(self) -> List[str]:
        """
        Load questions from file

        Returns:
            List of question strings
        """
        if not os.path.exists(self.questions_file):
            raise FileNotFoundError(f"Questions file not found: {self.questions_file}")

        with open(self.questions_file, 'r') as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data['questions'])} questions from {self.questions_file}")
        return data['questions']

    def save_response_cache(self, question_id: int, question: str, response_data: Dict[str, Any], provider_name: str) -> None:
        """
        Save response to cache file (per provider)

        Args:
            question_id: Unique identifier for the question
            question: The question text
            response_data: Dictionary containing response and metadata
            provider_name: Name of the provider (openai, gemini, claude)
        """
        cache_file = os.path.join(self.cache_dir, f"q_{question_id:04d}_{provider_name}.json")

        cache_data = {
            'question_id': question_id,
            'question': question,
            'provider': provider_name,
            'timestamp': datetime.now().isoformat(),
            'response': response_data.get('response'),
            'success': response_data.get('success'),
            'error': response_data.get('error'),
            'attempts': response_data.get('attempts')
        }

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

        logger.debug(f"Cached response for question {question_id} from {provider_name}")

    def load_response_cache(self, question_id: int, provider_name: str) -> Optional[Dict[str, Any]]:
        """
        Load response from cache for a specific provider

        Args:
            question_id: Unique identifier for the question
            provider_name: Name of the provider (openai, gemini, claude)

        Returns:
            Cached response data or None if not found
        """
        cache_file = os.path.join(self.cache_dir, f"q_{question_id:04d}_{provider_name}.json")

        if not os.path.exists(cache_file):
            return None

        with open(cache_file, 'r') as f:
            return json.load(f)

    def save_provider_results(self, provider_name: str, responses: List[Dict[str, Any]], output_file: str) -> None:
        """
        Save all responses from a specific provider to a dedicated file

        Args:
            provider_name: Name of the provider
            responses: List of all responses from this provider
            output_file: Path to output file (e.g., openai_responses.json)
        """
        successful = sum(1 for r in responses if r.get('success'))
        failed = len(responses) - successful

        data = {
            'provider': provider_name,
            'timestamp': datetime.now().isoformat(),
            'total_questions': len(responses),
            'successful': successful,
            'failed': failed,
            'responses': responses
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(responses)} responses from {provider_name} to {output_file}")

    def load_provider_results(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """
        Load results from a provider-specific file

        Args:
            provider_name: Name of the provider (openai, gemini, claude)

        Returns:
            Provider results dictionary or None if not found
        """
        filename = f"{provider_name}_responses.json"

        if not os.path.exists(filename):
            logger.warning(f"Provider results file not found: {filename}")
            return None

        with open(filename, 'r') as f:
            return json.load(f)

    def merge_provider_results(self, provider_names: List[str], total_questions: int, output_file: str = "results.json") -> Dict[str, Any]:
        """
        Merge results from all provider-specific files into one consolidated file

        Args:
            provider_names: List of provider names to merge
            total_questions: Total number of questions
            output_file: Path to output file

        Returns:
            Dictionary with consolidated results and statistics
        """
        # Load all provider results
        all_provider_data = {}
        for provider_name in provider_names:
            data = self.load_provider_results(provider_name)
            if data:
                all_provider_data[provider_name] = data

        if not all_provider_data:
            logger.error("No provider results found to merge")
            return {}

        # Initialize statistics
        stats = {
            'total_questions': total_questions,
            'by_provider': {},
            'questions_with_all_failed': [],
            'questions_with_partial_failures': []
        }

        # Build per-question consolidated results
        consolidated_results = []

        for q_id in range(total_questions):
            question_result = {
                'question_id': q_id,
                'question': None,
                'responses': {}
            }

            all_failed = True
            some_failed = False

            for provider_name, provider_data in all_provider_data.items():
                # Find this question's response
                response = next((r for r in provider_data['responses'] if r.get('question_id') == q_id), None)

                if response:
                    # Set question text (from any provider)
                    if question_result['question'] is None:
                        question_result['question'] = response.get('question')

                    # Store response
                    question_result['responses'][provider_name] = {
                        'response': response.get('response'),
                        'success': response.get('success'),
                        'error': response.get('error'),
                        'attempts': response.get('attempts')
                    }

                    if response.get('success'):
                        all_failed = False
                    else:
                        some_failed = True

            consolidated_results.append(question_result)

            # Track failure patterns
            if all_failed:
                stats['questions_with_all_failed'].append(q_id)
            elif some_failed:
                stats['questions_with_partial_failures'].append(q_id)

        # Calculate per-provider statistics
        for provider_name, provider_data in all_provider_data.items():
            stats['by_provider'][provider_name] = {
                'successful': provider_data.get('successful', 0),
                'failed': provider_data.get('failed', 0)
            }

        # Create consolidated output
        consolidated = {
            'generated_at': datetime.now().isoformat(),
            'statistics': stats,
            'results': consolidated_results
        }

        with open(output_file, 'w') as f:
            json.dump(consolidated, f, indent=2)

        logger.info(f"Merged results from {len(all_provider_data)} provider(s) to {output_file}")
        return consolidated

    def get_failed_questions(self, provider_name: Optional[str] = None) -> List[int]:
        """
        Get list of question IDs that failed for a specific provider

        Args:
            provider_name: Optional provider name to filter by

        Returns:
            List of question IDs
        """
        failed = set()

        for filename in os.listdir(self.cache_dir):
            if not filename.startswith('q_') or not filename.endswith('.json'):
                continue

            if provider_name:
                if f"_{provider_name}.json" not in filename:
                    continue

            cache_file = os.path.join(self.cache_dir, filename)
            with open(cache_file, 'r') as f:
                data = json.load(f)
                if not data.get('success', False):
                    failed.add(data['question_id'])

        return sorted(list(failed))

    def compare_responses(self, question_id: int, provider_names: List[str]) -> Dict[str, Any]:
        """
        Compare responses from different providers for a single question

        Args:
            question_id: Question ID to compare
            provider_names: List of provider names

        Returns:
            Dictionary with all responses for comparison
        """
        comparison = {
            'question_id': question_id,
            'question': None,
            'responses': {}
        }

        for provider_name in provider_names:
            cached = self.load_response_cache(question_id, provider_name)
            if cached:
                if comparison['question'] is None:
                    comparison['question'] = cached.get('question')

                comparison['responses'][provider_name] = {
                    'response': cached.get('response'),
                    'success': cached.get('success'),
                    'error': cached.get('error')
                }

        return comparison
