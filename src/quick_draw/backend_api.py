from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
from model import QuickDrawModel

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
model_path = os.getcwd()+"/models/quickdraw_model.pth"
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
    uvicorn.run(app, host="0.0.0.0", port=8000)