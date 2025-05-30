# CuentAI: AI-Powered Story Generator

CuentAI is an interactive web application that uses artificial intelligence to create personalized children's stories with automatically generated illustrations.

## Features

- **Story Generation**: Creates unique children's stories using OpenAI GPT-4
- **Illustration Generation**: Generates custom illustrations for each scene using DALL-E 3 or Stable Diffusion
- **Interactive UI**: User-friendly interface built with Streamlit
- **Customization**: Personalize the protagonist name and story theme
- **Multi-scene Stories**: View your story across multiple illustrated scenes
- **Optional Audio Narration**: Text-to-speech capabilities (requires Google Cloud setup)

## How It Works

1. Enter the protagonist's name
2. Choose a theme for the story
3. Click "Generate Story"
4. Enjoy your personalized story with AI-generated illustrations

## Technical Details

- **Backend**: Python with OpenAI API and Replicate API
- **Frontend**: Streamlit
- **Image Generation**: DALL-E 3 (OpenAI) and Stable Diffusion (Replicate)
- **Optional TTS**: Google Cloud Text-to-Speech

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CuentAI.git
cd CuentAI

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
# Create a .env file with your API keys:
# OPENAI_API_KEY=your_openai_api_key
# REPLICATE_API_TOKEN=your_replicate_api_token
```

## Usage

```bash
streamlit run app.py
```

## Requirements

See `requirements.txt` for a complete list of dependencies.

## License

MIT License

## Acknowledgements

- OpenAI for GPT-4 and DALL-E 3
- Replicate for Stable Diffusion access
- Streamlit for the web interface framework
