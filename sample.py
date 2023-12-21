import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 예시로 간단한 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(5, 3)

    def forward(self, x):
        return self.fc(x)

# 모델, 손실 함수, 옵티마이저 설정
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 예시 데이터 생성
inputs = torch.randn(100, 5)
targets = torch.randn(100, 3)

# 1000 epoch 동안 학습
total_epochs = 1000
steps_per_epoch = 10
total_iterations = total_epochs * steps_per_epoch

# tqdm을 사용하여 학습 진행 상황 시각화
for epoch in tqdm(range(total_epochs)):
    epoch_loss = 0.0

    # 한 번의 epoch에서 10 step씩 학습
    for step in tqdm(range(steps_per_epoch)):
        # 학습 진행
        optimizer.zero_grad()
        # 예시로, inputs와 targets는 데이터로 가정합니다.
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Loss 계산 및 출력
        epoch_loss += loss.item()

    # Epoch별 평균 Loss 출력
    epoch_loss /= steps_per_epoch
    print(f"Epoch [{epoch + 1}/{total_epochs}], Avg. Loss: {epoch_loss:.4f}")

# 학습 종료 후 추가 작업 수행
