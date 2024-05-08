## 🔨 Dacon 도배 하자 유형 분류 AI경진대회

### ✅ 요약
1. 주최 : 한솔데코
2. 주관 : 데이콘
3. 설명 : 총 19가지의 도배 하자 유형을 분류하는 AI 모델을 개발하여야 한다. 도배 하자 유형은 다음과 같다.<br>
(가구수정, 걸레받이수정, 곰팡이, 꼬임, 녹오염, 들뜸, 면불량, 몰딩수정, 문틀창틀수정, 반점, 석고수정, 오염, 오타공, 울음, 이음부불량, 터짐, 틈새과다, 피스, 훼손)
4. 결과<br>
**EfficientNet_b4** 사전 학습 모델을 불러와 추가 레이어를 쌓은 뒤 학습시킨 결과, **F1score 0.539**를 기록하였다.

---

### 📁 데이터셋
**1. train [폴더]**
- 19개의 Class 폴더 내 png 파일 존재

**2. test [폴더]**
- 평가용 데이터셋
- 000.png ~ 791.png

**3. test.csv [파일]**
- id : 평가 샘플 고유 id
- img_path : 평가 샘플의 이미지 파일 경로

**4. sample_submission.csv [제출양식]**
- id : 평가 샘플 고유 id  
- label : 예측한 도배 하자 Class

---

### 🔗 개발환경 (requirements.txt)

```
pandas==2.2.2
opencv-python==4.9.0.80
imbalanced-learn==0.12.2
torch
torchvision
torchaudio
albumentations
tqdm
```

---

### 📊 데이터 전처리 (processing.py)
**1. 이미지 경로 변경**
- train data의 img_path '\\' → '/'
- test data의 img_path './' → './data/'

**2. 한글 주소를 읽기 위한 encoding & decoding**

```python
img_array = np.fromfile(img_path, np.uint8)
image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
```

📌 클래스 불균형이 심하여 데이터 under/oversampling을 시도했으나, 성능이 오히려 낮아져서 적용하지 않았다.

---

### 📈 모델 선정 및 학습 (model.py, model_train.py)
**1. 모델 선정 : EfficientNet_b4**<br>
EfficientNet은 이미지 분류 작업에 있어서 효율적이고 우수한 성능을 보이는 딥러닝 모델이다. 네트워크의 깊이, 너비, 해상도의 크기를 조절하여 최적의 모델 구조를 찾아내는 compound scaling 방법을 적용하여 최소한의 파라미터로 높은 성능을 낸다. EfficientNet b0~b7 중 개발환경과 성능을 고려하여 b4를 선정하였다.

**2. label 형식 int → long 변환**

```python
labels = labels.long().to(device)
```

이 코드에서 라벨을 long 형식으로 변환하는 이유는 모델이 예상한 클래스 확률과 실제 클래스를 비교할 때 타입을 일치시키기 위해서다. PyTorch의 nn.CrossEntropyLoss() 손실 함수는 라벨을 long 형식으로 받기 때문에 정수 형식의 라벨을 long 형식으로 변환하여 손실 함수에 전달해야 한다. 또한, GPU를 사용하는 경우에는 모든 텐서가 동일한 타입을 가져야 하므로, GPU로 데이터를 이동하기 전에 라벨을 long 형식으로 변환하는 것이 중요하다.

---

### ✅ 결과
**👍리더보드(PUBLIC) : 275/1152(등)**

![image](https://github.com/2shin0/Papering-Flaw/assets/150658909/2f9504f0-a843-4bbf-a93a-31eae3dcc79c)
