import os
import openai
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import replicate
from dotenv import load_dotenv

# Page configuration must be the first Streamlit command
st.set_page_config(page_title="CuentAI – AI Story Maker", layout="wide")

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
replicate_token = os.getenv("REPLICATE_API_TOKEN")

# Optional TTS setup
# Uncomment if using Google Cloud TTS
# from google.cloud import texttospeech

# Function to load prompt templates
def load_prompt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        st.error(f"Error loading prompt file: {e}")
        return ""

# Story generation function
def generate_story(name: str, theme: str) -> str:
    """
    Prompt GPT-4 to write a 300–400 word children's story in English,
    with protagonist {name} and theme {theme}. Use playful tone,
    simple dialogue, and a clear beginning, middle, and end.
    """
    # Load prompt template
    prompt_template = load_prompt("prompts/story_prompt.txt")
    if not prompt_template:
        prompt_template = (
            f"You are a children's story author. "
            f"Write a 300-400 word children's story where the protagonist is named {name} "
            f"and the plot is about {theme}. Use a friendly style and simple dialogue."
        )
    
    # Format the prompt with user inputs
    prompt = prompt_template.format(name=name, theme=theme)
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating story: {e}")
        # Return a fallback story for demo purposes
        return f"""
        # The Great Discovery of {name}

        Once upon a time, there was a child named {name} who dreamed about {theme}.
        
        On a sunny day, {name} decided to explore the garden of their house. Among the flowers and trees, 
        they found a small door that had never been seen before.
        
        "What could this be?" {name} wondered curiously.
        
        Upon opening the door, they discovered a magical world full of bright colors and fantastic creatures.
        
        "Welcome!" said a talking butterfly. "We've been waiting for you."
        
        {name} spent the entire day meeting new friends and learning about the importance of caring for nature.
        
        When they returned home, they promised to come back soon and share their adventures with all their friends.
        
        The End.
        """

# Scene segmentation
def split_into_scenes(text: str, num_scenes: int = 3) -> list[str]:
    """
    Split story into specified number of scenes, trying to preserve paragraph structure.
    """
    # Split by paragraphs if possible, else chunk by word count
    paras = [p for p in text.split("\n") if p.strip()]
    
    if len(paras) >= num_scenes:
        # Combine paragraphs to get desired number of scenes
        result = []
        paragraphs_per_scene = len(paras) // num_scenes
        for i in range(num_scenes):
            start_idx = i * paragraphs_per_scene
            end_idx = start_idx + paragraphs_per_scene if i < num_scenes - 1 else len(paras)
            result.append("\n".join(paras[start_idx:end_idx]))
        return result
    else:
        # If not enough paragraphs, split by word count
        words = text.split()
        chunk_size = len(words) // num_scenes
        return [" ".join(words[i*chunk_size : (i+1)*chunk_size]) for i in range(num_scenes)]

# Image generation with DALL-E 3
def generate_image_dalle(prompt: str, protagonist: str) -> str:
    """
    Call OpenAI Image API to create one 512×512 image from the prompt.
    Returns the image URL.
    """
    # Load image prompt template
    img_prompt_template = load_prompt("prompts/image_prompt.txt")
    if not img_prompt_template:
        img_prompt_template = "Crea una ilustración de estilo infantil y colorido para un cuento para niños. La escena muestra: {scene_description} Con {protagonist_name} como personaje principal."
    
    # Format the prompt with user inputs
    full_prompt = img_prompt_template.format(
        scene_description=prompt,
        protagonist_name=protagonist
    )
    
    try:
        response = openai.Image.create(
            prompt=full_prompt,
            n=1,
            size="512x512"
        )
        return response["data"][0]["url"]
    except Exception as e:
        st.error(f"Error generating image: {e}")
        # Return a placeholder image URL
        return "https://via.placeholder.com/512x512.png?text=Image+Generation+Failed"

# Optional: Image generation with Replicate (Stable Diffusion)
def generate_image_replicate(prompt: str, protagonist: str) -> str:
    """
    Alternative image generation using Replicate API with Stable Diffusion.
    """
    if not replicate_token:
        st.warning("Replicate API token not set. Using fallback image.")
        return "https://via.placeholder.com/512x512.png?text=Replicate+API+Token+Missing"
    
    # Load image prompt template
    img_prompt_template = load_prompt("prompts/image_prompt.txt")
    if not img_prompt_template:
        img_prompt_template = "Crea una ilustración de estilo infantil y colorido para un cuento para niños. La escena muestra: {scene_description} Con {protagonist_name} como personaje principal."
    
    # Format the prompt with user inputs
    full_prompt = img_prompt_template.format(
        scene_description=prompt,
        protagonist_name=protagonist
    )
    
    try:
        client = replicate.Client(api_token=replicate_token)
        output = client.run(
            "stability-ai/sdxl:2b017d9b67edd2ee1401238df49d75da53c523f36e363881e057f5dc3ed3c5b2",
            input={"prompt": full_prompt}
        )
        if output and isinstance(output, list) and len(output) > 0:
            return output[0]
        else:
            raise Exception("No output from Replicate API")
    except Exception as e:
        st.error(f"Error generating image with Replicate: {e}")
        return "https://via.placeholder.com/512x512.png?text=Replicate+Image+Failed"

# Optional Audio TTS function
def generate_audio_tts(text: str, filename="narration.mp3") -> str:
    """
    Generate audio narration from text using Google Cloud TTS.
    """
    # Check if Google Cloud TTS is available
    try:
        from google.cloud import texttospeech
        
        # Load TTS prompt template
        tts_params = load_prompt("prompts/tts_prompt.txt")
        
        client = texttospeech.TextToSpeechClient()
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", 
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        response = client.synthesize_speech(
            input=input_text, 
            voice=voice, 
            audio_config=audio_config
        )
        
        with open(filename, "wb") as out:
            out.write(response.audio_content)
        
        return filename
    except ImportError:
        st.warning("Google Cloud Text-to-Speech is not installed. Skipping audio generation.")
        return None
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

# Custom CSS for child-friendly interface
def set_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@400;700&display=swap');
    
    * {
        font-family: 'Comic Neue', cursive;
    }
    
    h1, h2, h3 {
        color: #3366cc;
    }
    
    .stApp {
        background-color: #f0f8ff;
    }
    
    .stButton>button {
        background-color: #ff9966;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 10px 20px;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #ff7733;
    }
    
    .scene-container {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .story-text {
        font-size: 18px;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI
def main():
    # Apply custom CSS
    set_custom_css()
    
    # Header
    st.title("🧐‍♂️ CuentAI – AI Story Generator")
    st.markdown("### Create personalized stories with AI-generated images")
    
    # Sidebar with explanation
    with st.sidebar:
        st.subheader("About CuentAI")
        st.write("""
        CuentAI is an application that uses artificial intelligence to create personalized children's stories in English, 
        with automatically generated illustrations for each scene of the story.
        
        **How it works:**
        1. Enter the protagonist's name
        2. Choose a theme for the story
        3. Click on "Generate Story"
        4. Enjoy your personalized story with images!
        """)
        
        st.subheader("Technologies")
        st.write("""
        - OpenAI GPT-4 for generating text
        - DALL-E 3 for creating illustrations
        - Streamlit for the web interface
        """)
        
        # Image generation options
        st.subheader("Options")
        image_generator = st.radio(
            "Image generation engine:",
            options=["DALL-E 3", "Stable Diffusion (Replicate)"],
            index=0
        )
        st.session_state.image_generator = image_generator
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customize your story")
        protagonist = st.text_input("Protagonist Name", "Alice")
        theme = st.text_input("Story Theme", "explores a magical jungle")
        num_scenes = st.slider("Number of scenes", min_value=1, max_value=5, value=3)
        
        generate_button = st.button("✨ Generate Story")
        
        if generate_button:
            with st.spinner("Writing story with GPT-4..."):
                story_text = generate_story(protagonist, theme)
                st.session_state.story = story_text
                st.session_state.protagonist = protagonist
                st.session_state.scenes = split_into_scenes(story_text, num_scenes=num_scenes)
    
    with col2:
        if "story" not in st.session_state:
            st.image("https://img.freepik.com/free-vector/hand-drawn-fairy-tale-castle_23-2149423879.jpg", 
                    caption="Sample image - Generate your personalized story", 
                    use_column_width=True)
    
    # Display story and images
    if "story" in st.session_state:
        st.markdown("---")
        st.subheader("📚 Your Personalized Story")
        
        # Choose one layout: tabs, pagination, or scroll
        tabs = st.tabs([f"Scene {i+1}" for i in range(len(st.session_state.scenes))])
        
        for i, (tab, scene) in enumerate(zip(tabs, st.session_state.scenes)):
            with tab:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Generate image if not already in session state
                    if f"image_url_{i}" not in st.session_state:
                        with st.spinner("Generating illustration..."):
                            # Get first 100 words for the prompt to avoid token limits
                            scene_summary = " ".join(scene.split()[:100])
                            
                            # Use selected image generator
                            if st.session_state.image_generator == "DALL-E 3":
                                img_url = generate_image_dalle(scene_summary, st.session_state.protagonist)
                            else:
                                img_url = generate_image_replicate(scene_summary, st.session_state.protagonist)
                                
                            st.session_state[f"image_url_{i}"] = img_url
                    
                    # Display image
                    st.image(st.session_state[f"image_url_{i}"], use_column_width=True)
                    st.caption(f"Illustration generated for Scene {i+1}")
                
                with col2:
                    st.markdown(f"<div class='scene-container'><div class='story-text'>{scene}</div></div>", unsafe_allow_html=True)
        
        # Full story text
        with st.expander("View complete story"):
            st.markdown(f"<div class='story-text'>{st.session_state.story}</div>", unsafe_allow_html=True)
        
        # Optional TTS toggle
        st.markdown("---")
        st.subheader("🔊 Narration")
        
        if st.checkbox("Include audio narration"):
            # Check if TTS is imported
            try:
                from google.cloud import texttospeech
                with st.spinner("Generando audio..."):
                    if "audio_file" not in st.session_state:
                        audio_file = generate_audio_tts(st.session_state.story)
                        st.session_state.audio_file = audio_file
                    
                    if st.session_state.audio_file:
                        st.audio(st.session_state.audio_file)
                    else:
                        st.warning("Could not generate audio. Please check your Google Cloud configuration.")
            except ImportError:
                st.warning("""
                The narration feature requires Google Cloud Text-to-Speech.
                
                To enable this feature:
                1. Install the library: `pip install google-cloud-texttospeech`
                2. Configure your Google Cloud credentials
                """)
        
        # Download options
        st.markdown("---")
        st.subheader("💾 Save your story")
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="Download story text",
                data=st.session_state.story,
                file_name=f"story_{st.session_state.protagonist.lower().replace(' ', '_')}.txt",
                mime="text/plain"
            )
        
        # This is just a placeholder - in a real app you'd need to implement image downloading
        with col2:
            st.info("Image downloading will be available in a future version.")

# Run the app
if __name__ == "__main__":
    main()
