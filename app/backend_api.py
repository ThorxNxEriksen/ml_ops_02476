from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import sys


#How to run it 
# curl -X 'POST' \
#    'http://127.0.0.1:8000/predict' \
#    -H 'accept: application/json' \
#    -H 'Content-Type: multipart/form-data' \
#    -F 'file=@/mnt/c/Users/ThorNørgaardEriksen/OneDrive - Intellishore/3. Privat/2. Studie/2. Kurser/mlOps 02476/wall_test.png;type=image/png'


# Add src to sys.path, in order to properly import the QuickDrawModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from quick_draw.model import QuickDrawModel


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Categories
CATEGORIES = ['bear', 'broccoli', 'cake', 'cloud', 'bush', 
              'The Mona Lisa', 'The Great Wall of China', 
              'sea turtle', 'moustache', 'mouth']

# Load model
model = QuickDrawModel(num_classes=len(CATEGORIES))
model_path = "models/quickdraw_model.pth" #Update this to reflect if possible
print("THE FOLDER IS:", os.getcwd())
print("THE PATH IS: ",model_path)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

@app.get("/")
def read_root():
    return {"message": "QuickDraw API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('L')  # Convert to grayscale
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Get top prediction
        prob, idx = torch.max(probabilities, dim=0)
        predicted_category = CATEGORIES[idx.item()]
        confidence = prob.item()
    
    return {
        "category": predicted_category,
        "confidence": confidence
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))  # Use PORT from the environment, default to 8080
    uvicorn.run(app, host="0.0.0.0", port=port)
