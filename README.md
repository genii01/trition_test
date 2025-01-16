# Triton Inference Server with FasterRCNN Object Detection

FasterRCNN 모델을 Triton Inference Server로 서빙하는 객체 탐지 프로젝트입니다.

## 프로젝트 구조

```
.
├── client/
│   └── client_dev.py        # 개발용 클라이언트
├── model_repository/
│   └── fasterrcnn/         # 모델 저장소
├── Dockerfile              # Triton 서버 도커파일
├── triton_client.py       # 기본 클라이언트
└── model_format_conversion.py # 모델 변환기
```

## 빠른 시작

1. **환경 설정**
```bash
pip install -r requirements.txt
```

2. **모델 변환**
```bash
python model_format_conversion.py
```

3. **서버 실행**
```bash
docker build -t triton-fasterrcnn .
docker run --rm -p 8000-8002:8000-8002 triton-fasterrcnn
```

4. **클라이언트 실행**
```bash
python triton_client.py  # 기본 클라이언트
# 또는
python client/client_dev.py  # 개발용 클라이언트
```

## 주요 스펙

- **모델**: FasterRCNN ResNet50 FPN V2 (COCO 80클래스)
- **입력**: RGB 이미지 [3, H, W]
- **출력**: 
  - boxes: [N, 4] (FP32)
  - labels: [N] (INT64)
  - scores: [N] (FP32)
- **신뢰도 임계값**: 0.5

1. 기본 클라이언트:
```bash
python triton_client.py
```

2. 개발용 클라이언트:
```bash
python client/client_dev.py
```

## 주요 기능

- **모델**: FasterRCNN ResNet50 FPN V2
- **데이터셋**: COCO (80개 클래스 탐지)
- **기능**:
  - 실시간 객체 탐지
  - 바운딩 박스 시각화
  - 클래스 레이블 및 신뢰도 점수 표시
  - 50% 이상 신뢰도를 가진 객체만 표시

## 설정 상세

### Triton Server 설정
- 서버 포트:
  - HTTP: 8000
  - gRPC: 8001
  - Metrics: 8002
- 모델 플랫폼: PyTorch LibTorch
- 배치 크기: 1
- 실행 모드: CPU

### 입력/출력 형식
- 입력:
  - 이름: "input__0"
  - 형식: FP32
  - 차원: [3, -1, -1]

- 출력:
  - boxes: [N, 4] (FP32)
  - labels: [N] (INT64)
  - scores: [N] (FP32)

## 문제 해결

1. 서버 연결 오류
   - 포트(8000, 8001, 8002) 사용 가능 여부 확인
   - Docker 컨테이너 실행 상태 확인
   - 방화벽 설정 확인

2. 모델 로딩 오류
   - model.pt 파일 존재 여부 확인
   - 모델 변환 과정 재실행
   - 권한 설정 확인

3. 추론 결과 없음
   - 이미지 형식 확인 (RGB)
   - 입력 이미지 크기 확인
   - 신뢰도 임계값(0.5) 조정

## 참고 사항

- 모든 이미지는 RGB 형식으로 처리됩니다
- 신뢰도 임계값은 0.5로 설정되어 있습니다
- 기본적으로 CPU 모드로 실행됩니다
- config.pbtxt에서 GPU 설정 변경이 가능합니다
```

# Triton Inference Server 아키텍처

## 전체 아키텍처
```
[Client] <--HTTP/gRPC--> [Triton Server] <--> [Model Repository]
    |                           |
    |                           |
  Input                    Backend(PyTorch)
(train.png)                     |
                               |
                           Inference
                               |
                           Response
                        (boxes, labels, scores)
```

## 컴포넌트 설명

### 1. Client Layer
- **클라이언트 구현체**:
  - `triton_client.py`: 기본 클라이언트
  - `client/client_dev.py`: 개발용 클라이언트
- **통신 프로토콜**:
  - HTTP (8000)
  - gRPC (8001)
  - Metrics (8002)

### 2. Server Layer
```
[Triton Inference Server]
├── Model Repository
│   └── fasterrcnn/
│       ├── config.pbtxt    # 모델 설정
│       └── 1/              # 모델 버전
│           └── model.pt    # 모델 파일
├── Backend
│   └── PyTorch LibTorch
└── Dynamic Batching
```

### 3. 데이터 흐름
```
1. 이미지 입력
   [RGB Image] -> [Preprocessing] -> [Tensor(1,3,H,W)]

2. 서버 처리
   [Input Tensor] -> [FasterRCNN] -> [Raw Predictions]

3. 출력 처리
   [Raw Predictions] -> [Postprocessing] -> [Visualized Results]
```

## 인터페이스 상세

### 입력 스펙
```python
input {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [3, -1, -1]  # [채널, 높이, 너비]
}
```

### 출력 스펙
```python
output {
    "output__0": boxes    # shape: [N, 4], type: FP32
    "output__1": labels   # shape: [N], type: INT64
    "output__2": scores   # shape: [N], type: FP32
}
```

## 실행 흐름

1. **모델 준비**
   ```bash
   python model_format_conversion.py  # PyTorch -> TorchScript 변환
   ```

2. **서버 실행**
   ```bash
   docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 triton-fasterrcnn
   ```

3. **클라이언트 요청**
   ```python
   client = InferenceServerClient(url="localhost:8000")
   response = client.infer(model_name="fasterrcnn", inputs=inputs, outputs=outputs)
   ```

4. **결과 처리**
   ```python
   boxes = response.as_numpy("output__0")
   labels = response.as_numpy("output__1")
   scores = response.as_numpy("output__2")
   ```

## 성능 최적화

1. **Dynamic Batching**
   ```python
   dynamic_batching {
     preferred_batch_size: [1]
     max_queue_delay_microseconds: 100
   }
   ```

2. **Instance Group**
   ```python
   instance_group [
     {
       kind: KIND_CPU
       count: 1
     }
   ]
   ```

## 모니터링

- **Metrics 엔드포인트**: `localhost:8002/metrics`
- **모니터링 항목**:
  - 추론 요청 수
  - 처리 시간
  - 에러율
  - GPU 사용량 (GPU 모드)