"""Load the team's ResNet18 checkpoint and predict one PIL image."""
from pathlib import Path

import torch
from PIL import ImageOps
from torchvision.models import ResNet18_Weights, resnet18


def load_model(checkpoint_path):
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata["architecture"] != "resnet18" or metadata["preprocessing"] != "ResNet18_Weights.DEFAULT.transforms":
        raise ValueError("지원하지 않는 모델 또는 전처리 설정입니다.")
    classes = metadata["classes"]
    labels = [item["label"] for item in classes]
    if len(classes) != 10 or labels != list(range(10)):
        raise ValueError("모델의 품종 번호 순서를 확인해 주세요.")
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes, ResNet18_Weights.DEFAULT.transforms()


def predict_image(image, model, classes, transform):
    image = ImageOps.exif_transpose(image).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.inference_mode():
        values, indices = model(tensor).softmax(dim=1)[0].topk(3)
    return [
        {"breed": classes[index]["name"], "korean": classes[index]["korean"], "score": float(value)}
        for value, index in zip(values, indices.tolist())
    ]
