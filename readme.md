# Meeting Transcriber

Meeting Transcriber is a Python-based tool that automates the process of transcribing and summarizing audio and video files. It uses OpenAI's Whisper for transcription and various LLM APIs for summarization.

## Features

- Processes video files to extract audio
- Transcribes audio files using Whisper
- Summarizes transcripts using configurable LLM APIs
- Configurable settings via `config.yaml`
- Supports multiple LLM providers (OpenAI, Anthropic, Google, Replicate, Together AI)

## Installation

### System Dependencies

Before installing the Python packages, ensure you have the following system dependency:

- ffmpeg: Required for audio/video processing

#### Installing ffmpeg

- On Ubuntu or Debian:
  ```
  sudo apt update && sudo apt install ffmpeg
  ```

- On macOS (using Homebrew):
  ```
  brew install ffmpeg
  ```

- On Windows (using Chocolatey):
  ```
  choco install ffmpeg
  ```

### Python Dependencies

1. Clone the repository:
   ```
   git clone https://github.com/jfcostello/meeting-transcriber.git
   cd meeting-transcriber
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   GEMINI_API_KEY=your_gemini_api_key
   REPLICATE_API_KEY=your_replicate_api_key
   TOGETHERAI_API_KEY=your_togetherai_api_key
   ```

## Usage

1. Create folders within your `meeting_recordings_folder` (configured in config.yaml)
2. In each folder, create a `summary-rules.txt` file containing the system prompt for summarization
3. Place your video or audio files in the appropriate folders
4. Run the main script:
   ```
   python main.py
   ```
5. Check the output folders for transcripts and summaries

The system will:
- Recursively search through all subfolders
- Process files in any folder containing a summary-rules.txt file
- Use the content of summary-rules.txt as the system prompt for summarization
- Maintain folder structure in the output

## Folder Structure Example

```
meeting_recordings_folder/
├── ProjectA/
│   ├── summary-rules.txt
│   ├── meeting1.mp4
│   ├── meeting2.wav
│   └── SubProject/
│       ├── summary-rules.txt
│       └── discussion.mp3
└── ProjectB/
    ├── summary-rules.txt
    └── brainstorming.m4a
```

## Configuration

The `config.yaml` file allows you to customize various aspects of the transcription and summarization process. Here's a detailed breakdown of each configuration option:

### Folder Paths

- `meeting_recordings_folder`: The root folder where your organized recordings are stored
- `output_structure`: Configuration for how processed files are organized
  - `base_folder`: The root folder for all outputs
  - `structure`: List of subdirectory components (DATE, SUMMARY-TYPE, FILE-NAME)

### Transcription Engine Settings

- `transcription_engine`: The transcription engine to use. Options are "whisper" or "faster_whisper".

Faster Whisper is a faster and more efficient version of Whisper, which may be more suitable for larger audio files if you don't have a GPU.

#### Whisper Settings

- `model`: The Whisper model to use for transcription. Options include "tiny", "base", "small", "medium", and "large".
- `language`: The language of the audio. Set to "auto" for automatic detection or specify a language code.
- `device`: The device to use for processing. Options are "auto", "cpu", or "cuda".
- `batch_size`: The batch size for processing. Set to "auto" or specify an integer.
- `use_fp16`: Whether to use FP16 precision. Options are "auto", true, or false.
- `segment_length`: The length of audio segments to process. Set to "auto" or specify an integer (in seconds).

#### Faster Whisper Settings

- `model`: The Faster Whisper model to use for transcription. Options include "tiny", "base", "small", "medium", and "large".
- `device`: The device to use for processing. Options are "auto", "cpu", or "cuda".
- `compute_type`: The compute type for processing. Options are "float16", "int8_float16", or "int8".
- `beam_size`: The beam size for transcription.

#### WhisperX Settings

WhisperX provides enhanced transcription capabilities with speaker diarization and word-level alignment. Before using WhisperX, you'll need to:

1. Get a Hugging Face token from [Hugging Face](https://huggingface.co/settings/tokens)
2. Agree to the terms of service for:
   - [Segmentation](https://huggingface.co/pyannote/segmentation-3.0)
   - [Speaker-Diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

Note: The model runs entirely locally - the Hugging Face token is only required for accepting the terms of service.

Configuration options:

- `model`: The Whisper model to use. List as type of writing - tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, large-v3, large, distil-large-v2, distil-medium.en, distil-small.en, distil-large-v3, large-v3-turbo, turbo
- `device`: The processing device. Options are:
  - "auto": Automatically selects the best available device
  - "cpu": Forces CPU processing
  - "cuda": Uses GPU acceleration if available
- `compute_type`: The precision mode for processing. Options are:
  - "auto": Automatically selects the best available precision
  - "int8": 8-bit integer precision (fastest, least memory usage)
  - "int8_float16": Mixed precision for better accuracy
- `language`: The language of the audio. Set to "auto" for automatic detection or specify a language code (e.g., "en" for English).
- `batch_size`: Number of audio segments processed simultaneously. Adjust based on available GPU memory (default: 1).
- `diarize`: Enable speaker diarization to identify different speakers in the audio. Set to "true" or "false".
- `hf_token`: Your Hugging Face token for accessing speaker diarization models.
- `min_speakers`: (Optional) Minimum number of speakers to detect.
- `max_speakers`: (Optional) Maximum number of speakers to detect.
- `return_char_alignments`: Enable character-level alignment for precise word timestamps. Set to "true" or "false".
- `highlight_words`: Highlight words in the transcription with timestamps. Set to "true" or "false".

Accepted model

### LLM Settings

The LLM (Large Language Model) settings control how the summarization process works. These settings are crucial for determining which AI model will generate the summary and how it will behave.

- `model`: The name of the LLM model to use for summarization.
  - For OpenAI: e.g., "gpt-3.5-turbo", "gpt-4"
  - For Anthropic: e.g., "claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"
  - For Google: e.g., "gemini-pro"
  - For Replicate: Specify the full model string, e.g., "mistralai/mixtral-8x7b-instruct-v0.1"
  - For Together AI: e.g., "mistralai/Mixtral-8x7B-Instruct-v0.1"
  - For local models (e.g., with LM Studio): Use the model name specified in your local setup

- `client_type`: The type of LLM client to use. Options include:
  - "openai": For OpenAI's API
  - "anthropic": For Anthropic's API
  - "gemini": For Google's Gemini API
  - "replicate": For Replicate's API
  - "togetherai": For Together AI's API
  - "local_openai": For local models using the OpenAI-compatible API (e.g., LM Studio)

- `max_tokens`: The maximum number of tokens for the LLM response.
  - This limits the length of the generated summary.
  - Adjust based on your desired summary length and model capabilities.
  - Typical values range from 500 to 4000, depending on the model and use case.

- `temperature`: The temperature setting for the LLM (controls randomness).
  - Range is typically 0 to 1.
  - Lower values (e.g., 0.2) produce more focused, deterministic outputs.
  - Higher values (e.g., 0.8) produce more diverse, creative outputs.
  - For summarization, a lower temperature (0.2 - 0.5) is often preferred.

- `base_url`: (Optional) The base URL for the API endpoint.
  - Required when using `local_openai` client type or non-standard API endpoints.
  - For LM Studio, this would typically be "http://localhost:1234/v1" (adjust port as needed).

#### Using LM Studio with the Local OpenAI Approach

To use a local model with LM Studio:

1. Install and set up LM Studio on your machine.

2. In LM Studio, load your desired local model and start the local server.

3. In your `config.yaml`, set the following:

   ```yaml
   llm:
     model: "your-local-model-name"
     client_type: "local_openai"
     max_tokens: 2000
     temperature: 0.3
     base_url: "http://localhost:1234/v1"
   ```

   Replace "your-local-model-name" with the name of the model you're using in LM Studio, and adjust the port in `base_url` if necessary.

4. In your `.env` file, add:

   ```
   LOCAL_LLM_API_KEY=lm_studio
   ```

   LM Studio doesn't require an API key and uses this as a default

5. Run your script as usual. It will now use your local model through LM Studio for summarization.

#### Example Configurations

1. Using OpenAI's GPT-3.5:
   ```yaml
   llm:
     model: "gpt-3.5-turbo"
     client_type: "openai"
     max_tokens: 1000
     temperature: 0.3
   ```

2. Using Anthropic's Claude:
   ```yaml
   llm:
     model: "claude-3-5-sonnet-20240620"
     client_type: "anthropic"
     max_tokens: 2000
     temperature: 0.2
   ```

3. Using a local model with LM Studio:
   ```yaml
   llm:
     model: "llama-2-7b-chat"
     client_type: "local_openai"
     max_tokens: 1500
     temperature: 0.4
     base_url: "http://localhost:1234/v1"
   ```

Remember to ensure that you have the necessary API keys set in your `.env` file for the chosen `client_type`, except for `local_openai` which doesn't require an API key.

## Supported File Types

- Video: .mp4, .avi, .mov, .mkv
- Audio: .mp3, .wav, .m4a, .flac

## Troubleshooting

- If you encounter CUDA-related issues, ensure that your PyTorch installation matches your CUDA version.
- For any API-related errors, check that your API keys are correctly set in the `.env` file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.