FROM nvcr.io/nvidia/tritonserver:23.12-py3

# 작업 디렉토리 설정
WORKDIR /opt/tritonserver

# 작업 디렉토리에 있는 모델 파일을 컨테이너 내부로 복사
COPY ./model.pt /opt/tritonserver/models/fasterrcnn/1/model.pt

# 필요한 경우 권한 설정
RUN chmod 644 /opt/tritonserver/models/fasterrcnn/1/model.pt

# 모델 설정 파일 복사
COPY model_repository/fasterrcnn/config.pbtxt /opt/tritonserver/models/fasterrcnn/config.pbtxt

# Triton Server 포트 노출
EXPOSE 8000 8001 8002

# Triton Server 실행
CMD ["tritonserver", "--model-repository=/opt/tritonserver/models", "--log-verbose=1"]