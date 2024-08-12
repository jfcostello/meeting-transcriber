import os
from llm_utils import call_llm_api
from config_handler import get_config

def summarize_transcript(transcript_path, config):
    file_name = os.path.splitext(os.path.basename(transcript_path))[0]
    output_folder = config['summaries_folder']
    output_path = os.path.join(output_folder, f"{file_name}_summary.md")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Read transcript
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    # Get LLM configuration
    llm_config = config['llm']
    prompt_template = get_config()['prompts'].get(config['summary_prompt'], "Summarize the following transcript:")

    # Prepare prompt
    full_prompt = f"{prompt_template}\n\n{transcript}"

    # Call LLM API
    summary = call_llm_api(
        model=llm_config['model'],
        content=full_prompt,
        systemPrompt=llm_config['system_prompt'],
        max_tokens=llm_config['max_tokens'],
        temperature=llm_config['temperature'],
        client_type=llm_config['client_type']
    )

    # Save summary as markdown
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Summary: {file_name}\n\n")
        f.write(summary)

    print(f"Summary saved: {output_path}")
    return output_path