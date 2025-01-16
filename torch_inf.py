import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_model():
    # 최신 방식으로 모델 로드
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    model.eval()

    # 모델 저장
    torch.save(model.state_dict(), "fasterrcnn_model.pth")
    return model


def load_saved_model():
    # 저장된 모델 불러오기
    model = fasterrcnn_resnet50_fpn_v2(pretrained=False)
    model.load_state_dict(torch.load("fasterrcnn_model.pth"))
    model.eval()
    return model


def detect_objects(image_path, model):
    # 이미지를 RGB 모드로 불러오기
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)

    # COCO 데이터셋의 클래스 이름
    COCO_CLASSES = [
        "N/A",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "N/A",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "N/A",
        "backpack",
        "umbrella",
        "N/A",
        "N/A",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "N/A",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "N/A",
        "dining table",
        "N/A",
        "N/A",
        "toilet",
        "N/A",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "N/A",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    # 객체 탐지 수행
    with torch.no_grad():
        prediction = model([image_tensor])

        # prediction 결과를 보기 쉽게 출력
        boxes = prediction[0]["boxes"]
        labels = prediction[0]["labels"]
        scores = prediction[0]["scores"]

        # 가장 높은 신뢰도를 가진 객체 찾기
        max_score_idx = scores.argmax().item()
        max_score = scores[max_score_idx].item()

        if max_score > 0.5:  # 신뢰도 50% 이상인 경우만 출력
            print("\n탐지된 객체 (최고 신뢰도):")
            print(
                f"객체: {COCO_CLASSES[labels[max_score_idx]]}, 신뢰도: {max_score:.2f}"
            )

    # 결과 시각화
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # 탐지된 객체에 박스 그리기
    for box, label, score in zip(
        prediction[0]["boxes"], prediction[0]["labels"], prediction[0]["scores"]
    ):
        if score > 0.5:  # 신뢰도가 50% 이상인 경우만 표시
            x1, y1, x2, y2 = box.numpy()
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none"
            )
            ax.add_patch(rect)
            plt.text(
                x1,
                y1,
                f"{COCO_CLASSES[label]}: {score:.2f}",
                bbox=dict(facecolor="white", alpha=0.8),
            )

    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # 새로운 모델 다운로드 및 저장
    model = load_model()

    # 또는 저장된 모델 불러오기
    # model = load_saved_model()

    # 이미지에서 객체 탐지
    image_path = "./train.png"  # 테스트할 이미지 경로
    detect_objects(image_path, model)
