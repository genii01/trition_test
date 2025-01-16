import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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


def preprocess_image(image_path):
    # 이미지 전처리
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)
    # 배치 차원 추가 (1, C, H, W)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor.numpy(), image


def visualize_results(image, boxes, labels, scores):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # 신뢰도 50% 이상만 표시
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


def main():
    try:
        # Triton 클라이언트 생성
        client = InferenceServerClient(url="localhost:8000")

        # 이미지 로드 및 전처리
        image_path = "train.png"
        input_data, original_image = preprocess_image(image_path)

        # 입력 설정
        inputs = []
        inputs.append(
            InferInput(
                "input__0", input_data.shape, np_to_triton_dtype(input_data.dtype)
            )
        )
        inputs[0].set_data_from_numpy(input_data)

        # 출력 설정
        outputs = []
        outputs.append(InferRequestedOutput("output__0"))  # boxes
        outputs.append(InferRequestedOutput("output__1"))  # labels
        outputs.append(InferRequestedOutput("output__2"))  # scores

        # 추론 실행 (model_version 파라미터 수정)
        response = client.infer(model_name="fasterrcnn", inputs=inputs, outputs=outputs)

        # 결과 처리
        boxes = response.as_numpy("output__0")
        labels = response.as_numpy("output__1")
        scores = response.as_numpy("output__2")

        # 결과 시각화
        visualize_results(original_image, boxes, labels, scores)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
