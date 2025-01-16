import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import transforms
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput


def load_image(image_path):
    # 이미지를 RGB 모드로 불러오기
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)

    # numpy로 변환하고 batch dimension 추가
    return image, image_tensor.numpy()[None, ...]


def detect_objects_triton(image_path, triton_url="localhost:8000"):
    # Triton 클라이언트 설정
    client = InferenceServerClient(url=triton_url)

    # 이미지 로드 및 전처리
    original_image, image_data = load_image(image_path)

    # 입력 설정
    input_tensor = InferInput(name="images", shape=image_data.shape, datatype="FP32")
    input_tensor.set_data_from_numpy(image_data)

    # 추론 요청
    results = client.infer(model_name="fasterrcnn", inputs=[input_tensor])

    # 결과 파싱
    boxes = results.as_numpy("boxes")
    scores = results.as_numpy("scores")
    labels = results.as_numpy("labels")

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
    ]  # 전체 클래스 리스트는 기존 코드와 동일하게 유지

    # 결과 시각화
    fig, ax = plt.subplots(1)
    ax.imshow(original_image)

    # 탐지된 객체에 박스 그리기
    for box, label, score in zip(boxes[0], labels[0], scores[0]):
        if score > 0.5:  # 신뢰도가 50% 이상인 경우만 표시
            x1, y1, x2, y2 = box
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
    image_path = "./train.png"  # 테스트할 이미지 경로
    detect_objects_triton(image_path)
