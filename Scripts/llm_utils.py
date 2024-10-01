import os
from dotenv import load_dotenv

load_dotenv()

def call_llm_api(model, content, systemPrompt, max_tokens=4000, temperature=0, client_type="default", base_url=None):
    if client_type == "openai" or client_type == "local_openai":
        from openai import OpenAI
        if client_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        else:
            api_key = os.getenv("LOCAL_LLM_API_KEY", "not-needed")
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": content}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response_content = response.choices[0].message.content
        return response_content
    elif client_type == "groq":
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        chat_completion = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": content},
                {"role": "system", "content": systemPrompt}
            ]
        )
        response_content = chat_completion.choices[0].message.content
        return response_content
    elif client_type == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        chat_completion = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=systemPrompt,
            messages=[{"role": "user", "content": content}]
        )
        try:
            response_content = chat_completion.content[0].text
            return response_content
        except (IndexError, AttributeError) as e:
            raise ValueError(f"Error parsing Anthropic response: {e}")
    elif client_type == "gemini":
        from google.generativeai import GenerativeModel
        from google.generativeai import configure
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        configure(api_key=os.environ["GEMINI_API_KEY"])

        generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain",
        }

        gemini_model = GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            system_instruction=systemPrompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        chat_session = gemini_model.start_chat()
        chat_completion = chat_session.send_message(content)

        try:
            response_content = chat_completion.candidates[0].content.parts[0].text
            return response_content
        except (IndexError, AttributeError) as e:
            raise ValueError(f"Error parsing Gemini response: {e}")
    elif client_type == "replicate":
        import replicate
        client = replicate.Client(api_token=os.getenv("REPLICATE_API_KEY"))
        output = client.run(
            model,
            input={
                "top_k": 0,
                "top_p": 0.95,
                "prompt": content,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system_prompt": systemPrompt,
                "presence_penalty": 0,
                "log_performance_metrics": False
            }
        )
        response_content = "".join(output)
        return response_content
    elif client_type == "togetherai":
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("TOGETHERAI_API_KEY"),
            base_url="https://api.together.xyz/v1",
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": content},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        response_content = response.choices[0].message.content
        return response_content
    else:
        raise ValueError(f"Unsupported client type: {client_type}")
    