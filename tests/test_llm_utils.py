import unittest
from unittest.mock import patch
from Scripts.llm_utils import call_llm_api
import os

class TestLLMUtils(unittest.TestCase):
    @patch('Scripts.llm_utils.OpenAI')
    def test_call_llm_api_openai(self, mock_openai):
        # BDD:
        #   Scenario: Call LLM API with OpenAI
        #     Given a model, content, system prompt, and client type of 'openai'
        #     When the call_llm_api function is called
        #     Then the OpenAI API should be called
        #     And the function should return the response content
        # Pass Criteria:
        #   The OpenAI API is called, and the function returns the response content.
        mock_client = mock_openai.return_value
        mock_response = mock_client.chat.completions.create.return_value
        mock_response.choices = [{"message": {"content": "Test OpenAI response"}}]
        
        response = call_llm_api(model="gpt-3.5-turbo", content="Test content", systemPrompt="Test system prompt", client_type="openai")
        self.assertEqual(response, "Test OpenAI response")
        mock_client.chat.completions.create.assert_called_once()

    @patch('Scripts.llm_utils.Groq')
    def test_call_llm_api_groq(self, mock_groq):
        # BDD:
        #   Scenario: Call LLM API with Groq
        #     Given a model, content, system prompt, and client type of 'groq'
        #     When the call_llm_api function is called
        #     Then the Groq API should be called
        #     And the function should return the response content
        # Pass Criteria:
        #   The Groq API is called, and the function returns the response content.
        mock_client = mock_groq.return_value
        mock_response = mock_client.chat.completions.create.return_value
        mock_response.choices = [{"message": {"content": "Test Groq response"}}]
        
        response = call_llm_api(model="mixtral-8x7b-32768", content="Test content", systemPrompt="Test system prompt", client_type="groq")
        self.assertEqual(response, "Test Groq response")
        mock_client.chat.completions.create.assert_called_once()

    @patch('Scripts.llm_utils.Anthropic')
    def test_call_llm_api_anthropic(self, mock_anthropic):
        # BDD:
        #   Scenario: Call LLM API with Anthropic
        #     Given a model, content, system prompt, and client type of 'anthropic'
        #     When the call_llm_api function is called
        #     Then the Anthropic API should be called
        #     And the function should return the response content
        # Pass Criteria:
        #   The Anthropic API is called, and the function returns the response content.
        mock_client = mock_anthropic.return_value
        mock_response = mock_client.messages.create.return_value
        mock_response.content = [{"text": "Test Anthropic response"}]
        
        response = call_llm_api(model="claude-3-opus-20240229", content="Test content", systemPrompt="Test system prompt", client_type="anthropic")
        self.assertEqual(response, "Test Anthropic response")
        mock_client.messages.create.assert_called_once()

    @patch('Scripts.llm_utils.GenerativeModel')
    def test_call_llm_api_gemini(self, mock_gemini):
        # BDD:
        #   Scenario: Call LLM API with Gemini
        #     Given a model, content, system prompt, and client type of 'gemini'
        #     When the call_llm_api function is called
        #     Then the Gemini API should be called
        #     And the function should return the response content
        # Pass Criteria:
        #   The Gemini API is called, and the function returns the response content.
        mock_model = mock_gemini.return_value
        mock_session = mock_model.start_chat.return_value
        mock_response = mock_session.send_message.return_value
        mock_response.candidates = [{"content": {"parts": [{"text": "Test Gemini response"}]}}]

        response = call_llm_api(model="gemini-pro", content="Test content", systemPrompt="Test system prompt", client_type="gemini")
        self.assertEqual(response, "Test Gemini response")
        mock_session.send_message.assert_called_once()

    @patch('Scripts.llm_utils.replicate.Client')
    def test_call_llm_api_replicate(self, mock_replicate):
        # BDD:
        #   Scenario: Call LLM API with Replicate
        #     Given a model, content, system prompt, and client type of 'replicate'
        #     When the call_llm_api function is called
        #     Then the Replicate API should be called
        #     And the function should return the response content
        # Pass Criteria:
        #   The Replicate API is called, and the function returns the response content.
        mock_client = mock_replicate.return_value
        mock_client.run.return_value = ["Test Replicate response"]

        response = call_llm_api(model="meta/llama-2-70b-chat", content="Test content", systemPrompt="Test system prompt", client_type="replicate")
        self.assertEqual(response, "Test Replicate response")
        mock_client.run.assert_called_once()

    @patch('Scripts.llm_utils.OpenAI')
    def test_call_llm_api_togetherai(self, mock_openai):
        # BDD:
        #   Scenario: Call LLM API with TogetherAI
        #     Given a model, content, system prompt, and client type of 'togetherai'
        #     When the call_llm_api function is called
        #     Then the TogetherAI API should be called
        #     And the function should return the response content
        # Pass Criteria:
        #   The TogetherAI API is called, and the function returns the response content.
        mock_client = mock_openai.return_value
        mock_response = mock_client.chat.completions.create.return_value
        mock_response.choices = [{"message": {"content": "Test TogetherAI response"}}]

        response = call_llm_api(model="mistralai/Mixtral-8x7B-Instruct-v0.1", content="Test content", systemPrompt="Test system prompt", client_type="togetherai")
        self.assertEqual(response, "Test TogetherAI response")
        mock_client.chat.completions.create.assert_called_once()

    def test_call_llm_api_unsupported_client(self):
        # BDD:
        #   Scenario: Call LLM API with unsupported client
        #     Given a model, content, system prompt, and an unsupported client type
        #     When the call_llm_api function is called
        #     Then the function should raise a ValueError
        # Pass Criteria:
        #   The function raises a ValueError when called with an unsupported client type.
        with self.assertRaises(ValueError):
            call_llm_api(model="test_model", content="Test content", systemPrompt="Test system prompt", client_type="unsupported")

if __name__ == '__main__':
    unittest.main()
