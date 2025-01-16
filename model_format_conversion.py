import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2


class FasterRCNNWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        predictions = self.model(x)[0]
        return (predictions["boxes"], predictions["labels"], predictions["scores"])


def convert_model():
    # 모델 로드
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    model.eval()

    # 래퍼 모델 생성
    wrapped_model = FasterRCNNWrapper(model)

    # TorchScript로 변환
    example_input = torch.randn(1, 3, 224, 224)
    scripted_model = torch.jit.trace(wrapped_model, [example_input])

    # 모델 저장
    scripted_model.save("model.pt")


if __name__ == "__main__":
    convert_model()
