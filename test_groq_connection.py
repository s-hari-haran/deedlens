
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
import groq
import base64

# Load env vars
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ocr.ocr_engine import OCREngine, OCRBackend

def create_test_image():
    """Create a simple image with text for testing."""
    img = Image.new('RGB', (800, 200), color='white')
    d = ImageDraw.Draw(img)
    
    # Add text
    text = "This is a test document for DeedLens.\nSurvey No. 1234\nDate: 15-01-2024"
    d.text((50, 50), text, fill=(0, 0, 0))
    
    img_path = "test_ocr_image.jpg"
    img.save(img_path)
    return img_path

def test_groq():
    print("Testing Groq OCR with Llama 4 Scout...")
    
    # Check API key
    key = os.getenv("GROQ_API_KEY")
    client = groq.Groq(api_key=key)
    
    # Create test image
    img_path = create_test_image()
    
    # Convert image to base64
    with open(img_path, "rb") as image_file:
        img_str = base64.b64encode(image_file.read()).decode('utf-8')
    
    model = "meta-llama/llama-4-scout-17b-16e-instruct"
    print(f"Trying model: {model}...")
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text from this image:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_str}",
                            },
                        },
                    ],
                }
            ],
            model=model,
            temperature=0.0,
        )
        
        text = chat_completion.choices[0].message.content
        print("\n--- Result ---")
        print(f"Text: [{text}]")
        
    except Exception as e:
        print(f"Failed: {str(e)}")
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

def list_models():
    print("Listing Groq models...")
    key = os.getenv("GROQ_API_KEY")
    client = groq.Groq(api_key=key)
    try:
        models = client.models.list()
        print("Vision Models:")
        for m in models.data:
            if "vision" in m.id.lower():
                print(f"- {m.id}")
        
        print("\nAll Models:")
        for m in models.data:
             print(f"- {m.id}")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    # list_models()
    test_groq()
