from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from io import BytesIO
import base64
import subprocess
from gtts import gTTS
import os
import tempfile
import re

app = FastAPI()

# Enable CORS if needed.
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request and Response models.
class ChatRequest(BaseModel):
    query: str

class TalkingResponse(BaseModel):
    text: str
    audio_base64: str  # Audio (e.g., MP3) encoded in base64.
    video_base64: str  # Generated talking video (MP4) encoded in base64.

# Helper function to call your model (using Ollama in your case).
def run_ollama_model(query: str) -> str:
    model_name = "hf.co/lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF:Q3_K_L"
    command = ["ollama", "run", model_name]
    try:
        result = subprocess.run(
            command,
            input=query,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise Exception(f"Ollama CLI error: {e.stderr}")
    return result.stdout.strip()

# Function to generate the talking video
def generate_talking_video(audio_bytes: bytes, reference_video_path: str, checkpoint_path: str) -> str:
    """
    Generates a talking head video using the Wav2Lip model.
    
    Parameters:
      - audio_bytes: Audio data in bytes.
      - reference_video_path: Path to the reference video of the woman.
      - checkpoint_path: Path to the pretrained Wav2Lip checkpoint.
    
    Returns:
      - video_base64: The generated video (MP4) encoded in base64.
    """
    # Write the audio bytes to a temporary MP3 file.
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_temp:
        audio_temp.write(audio_bytes)
        audio_file_path = audio_temp.name

    # Create a temporary file path for the output video.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_temp:
        output_video_path = video_temp.name

    # Build the command to run Wav2Lip.
    command = [
        "python", "Wav2Lip/inference.py",
        "--checkpoint_path", checkpoint_path,
        "--face", reference_video_path,
        "--audio", audio_file_path,
        "--outfile", output_video_path
    ]

    try:
        # Run the command and capture both stdout and stderr.
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Wav2Lip Output: {result.stdout.decode()}")
        print(f"Wav2Lip Error: {result.stderr.decode()}")
    except subprocess.CalledProcessError as e:
        # Clean up the temporary audio file.
        os.remove(audio_file_path)
        raise Exception(f"Wav2Lip inference failed. Error: {e.stderr.decode()}")

    # Read the generated video and encode it in base64.
    with open(output_video_path, "rb") as f:
        video_bytes = f.read()
    video_base64 = base64.b64encode(video_bytes).decode("utf-8")

    # Clean up the temporary files.
    os.remove(audio_file_path)
    os.remove(output_video_path)

    return video_base64

@app.post("/talking", response_model=TalkingResponse)
async def talking_endpoint(request: ChatRequest):
    query = request.query

    # 1. Get the text response from your model.
    try:
        text_response = run_ollama_model(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference error: {str(e)}")

    # 2. Convert the text response to audio using gTTS.
    # try:
    #     text_response = re.sub(r"<think>.*?</think>\s*", "", text_response, flags=re.DOTALL)
    #     print (text_response)
    #     tts = gTTS(text_response, lang="en")
    #     audio_fp = BytesIO()
    #     tts.write_to_fp(audio_fp)
    #     audio_fp.seek(0)
    #     audio_bytes = audio_fp.read()
    #     audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"TTS conversion error: {str(e)}")
    try:
        text_response = re.sub(r"<think>.*?</think>\s*", "", text_response, flags=re.DOTALL)
        print(text_response)

        tts = gTTS(text_response, lang="en")
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        audio_bytes = audio_fp.read()

        # Save the original TTS audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        # Create a new temporary file for the male voice version
        temp_male_audio_path = temp_audio_path.replace(".mp3", "_male.mp3")

        # Use SoX to lower the pitch (-300 shifts down to a deeper male voice)
        command = [
            "sox", temp_audio_path, temp_male_audio_path, "pitch", "-300"
        ]

        # Run the SoX command
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"SoX error: {e.stderr.decode()}")

        # Read the modified (male) audio file
        with open(temp_male_audio_path, "rb") as f:
            male_audio_bytes = f.read()
        audio_base64 = base64.b64encode(male_audio_bytes).decode("utf-8")

        # Cleanup temporary files
        #os.remove(temp_audio_path)
        #os.remove(temp_male_audio_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS conversion error: {str(e)}")


    # 3. Generate the talking video using Wav2Lip.
    reference_video_path = "/Users/ahsanhussain/Documents/python/Ai/TTS/video4.mp4"  # Replace with your actual file path.
    checkpoint_path = "/Users/ahsanhussain/Documents/python/Ai/TTS/Wav2lip/checkpoints/wav2lip_gan.pth"      # Replace with your checkpoint path.
    try:
        video_base64 = generate_talking_video(audio_bytes, reference_video_path, checkpoint_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video creation error: {str(e)}")

    return TalkingResponse(
        text=text_response,
        audio_base64=audio_base64,
        video_base64=video_base64
    )
