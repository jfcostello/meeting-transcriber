import unittest
from unittest.mock import patch
from Scripts.llm_utils import call_llm_api
import os

class TestLLMUtils(unittest.TestCase):
    @patch('Scripts.llm_utils.OpenAI')
    def test_call_llm_api_openai(self, mock_openai):
        # BDD: Scenario: Using different LLM models - Then the transcript is summarized using the specified LLM model
        # Test: Checks if the OpenAI API is called correctly
        mock_client = mock_openai.return_value
        mock_response = mock_client.chat.completions.create.return_value
        mock_response.choices = [{"message": {"content": "Test OpenAI response"}}]
        
        response = call_llm_api(model="gpt-3.5-turbo", content="Test content", systemPrompt="Test system prompt", client_type="openai")
        self.assertEqual(response, "Test OpenAI response")
        mock_client.chat.completions.create.assert_called_once()

    @patch('Scripts.llm_utils.Groq')
    def test_call_llm_api_groq(self, mock_groq):
        # BDD: Scenario: Using different LLM models - Then the transcript is summarized using the specified LLM model
        # Test: Checks if the Groq API is called correctly
        mock_client = mock_groq.return_value
        mock_response = mock_client.chat.completions.create.return_value
        mock_response.choices = [{"message": {"content": "Test Groq response"}}]
        
        response = call_llm_api(model="mixtral-8x7b-32768", content="Test content", systemPrompt="Test system prompt", client_type="groq")
        self.assertEqual(response, "Test Groq response")
        mock_client.chat.completions.create.assert_called_once()

    @patch('Scripts.llm_utils.Anthropic')
    def test_call_llm_api_anthropic(self, mock_anthropic):
        # BDD: Scenario: Using different LLM models - Then the transcript is summarized using the specified LLM model
        # Test: Checks if the Anthropic API is called correctly
        mock_client = mock_anthropic.return_value
        mock_response = mock_client.messages.create.return_value
        mock_response.content = [{"text": "Test Anthropic response"}]
        
        response = call_llm_api(model="claude-3-opus-20240229", content="Test content", systemPrompt="Test system prompt", client_type="anthropic")
        self.assertEqual(response, "Test Anthropic response")
        mock_client.messages.create.assert_called_once()

    @patch('Scripts.llm_utils.GenerativeModel')
    def test_call_llm_api_gemini(self, mock_gemini):
        # BDD: Scenario: Using different LLM models - Then the transcript is summarized using the specified LLM model
        # Test: Checks if the Gemini API is called correctly
        mock_model = mock_gemini.return_value
        mock_session = mock_model.start_chat.return_value
        mock_response = mock_session.send_message.return_value
        mock_response.candidates = [{"content": {"parts": [{"text": "Test Gemini response"}]}}]

        response = call_llm_api(model="gemini-pro", content="Test content", systemPrompt="Test system prompt", client_type="gemini")
        self.assertEqual(response, "Test Gemini response")
        mock_session.send_message.assert_called_once()

    @patch('Scripts.llm_utils.replicate.Client')
    def test_call_llm_api_replicate(self, mock_replicate):
        # BDD: Scenario: Processing a file with a Replicate model - Then the transcript is summarized using the Replicate model
        # Test: Checks if the Replicate API is called correctly
        mock_client = mock_replicate.return_value
        mock_client.run.return_value = ["Test Replicate response"]

        response = call_llm_api(model="meta/llama-2-70b-chat", content="Test content", systemPrompt="Test system prompt", client_type="replicate")
        self.assertEqual(response, "Test Replicate response")
        mock_client.run.assert_called_once()

    @patch('Scripts.llm_utils.OpenAI')
    def test_call_llm_api_togetherai(self, mock_openai):
        # BDD: Scenario: Processing a file with a TogetherAI model - Then the transcript is summarized using the TogetherAI model
        # Test: Checks if the TogetherAI API is called correctly
        mock_client = mock_openai.return_value
        mock_response = mock_client.chat.completions.create.return_value
        mock_response.choices = [{"message": {"content": "Test TogetherAI response"}}]

        response = call_llm_api(model="mistralai/Mixtral-8x7B-Instruct-v0.1", content="Test content", systemPrompt="Test system prompt", client_type="togetherai")
        self.assertEqual(response, "Test TogetherAI response")
        mock_client.chat.completions.create.assert_called_once()

    def test_call_llm_api_unsupported_client(self):
        # BDD: N/A
        # Test: Checks if an error is raised for an unsupported client type
        with self.assertRaises(ValueError):
            call_llm_api(model="test_model", content="Test content", systemPrompt="Test system prompt", client_type="unsupported")

if __name__ == '__main__':
    unittest.main()
