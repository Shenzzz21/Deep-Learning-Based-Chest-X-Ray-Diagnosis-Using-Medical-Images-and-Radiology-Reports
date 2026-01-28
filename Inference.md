## Inference Using the Trained Model (.pth)

A pretrained multimodal model checkpoint (`.pth`) is provided. This allows inference on new chest X-ray images and clinical text without retraining.

---

### Prerequisites
- Python 3.8 or higher  
- PyTorch  
- Torchvision  
- Hugging Face Transformers  
- Pillow  
- Gradio (optional)

Install dependencies:
```bash
pip install torch torchvision transformers pillow gradio
```

Loading the Model:
```bash
import torch
from model import MultimodalModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultimodalModel(num_classes=2)
model.load_state_dict(
    torch.load("final_best_multimodal_xray.pth", map_location=device)
)
model.to(device)
model.eval()
```

Preparing Inputs:
```bash
from torchvision import transforms
from PIL import Image

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open("example_xray.png").convert("RGB")
image_tensor = image_transform(image).unsqueeze(0).to(device)
```

Text Input:
```bash
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

text = "No acute cardiopulmonary abnormality identified."
encoding = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=256,
    return_tensors="pt"
)

input_ids = encoding["input_ids"].to(device)
attention_mask = encoding["attention_mask"].to(device)
```

Running Inference:
```bash
with torch.no_grad():
    outputs = model(image_tensor, input_ids, attention_mask)
    probabilities = torch.softmax(outputs, dim=1)

normal_prob = probabilities[0][0].item()
abnormal_prob = probabilities[0][1].item()

print(f"Normal: {normal_prob:.2f}, Abnormal: {abnormal_prob:.2f}")
```

### Interpreting the Output
- The predicted class is the one with the higher probability
- Probabilities indicate model confidence
- This system is intended as a decision-support tool, not a standalone diagnostic system

### Optional: Web-Based Inference

A Gradio-based web interface is included in the project. Running the main script will launch an interactive application that allows users to upload a chest X-ray image and enter clinical text for real-time predictions.
