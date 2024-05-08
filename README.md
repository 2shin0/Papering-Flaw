## Dacon 도배 하자 유형 분류 AI경진대회

### ✅ 안내
>1. 주최 : 한솔데코
>2. 주관 : 데이콘
>3. 설명 : 총 19가지의 도배 하자 유형을 분류하는 AI 모델을 개발하여야 한다. 도배 하자 유형은 다음과 같다.<br>
>(가구수정, 걸레받이수정, 곰팡이, 꼬임, 녹오염, 들뜸, 면불량, 몰딩수정, 문틀창틀수정, 반점, 석고수정, 오염, 오타공, 울음, 이음부불량, 터짐, 틈새과다, 피스, 훼손)

---

### ✅ 데이터셋 설명
1. train [폴더]
>19개의 Class 폴더 내 png 파일 존재

2. test [폴더]
>평가용 데이터셋
>000.png ~ 791.png

3. test.csv [파일]
>id : 평가 샘플 고유 id
>img_path : 평가 샘플의 이미지 파일 경로

4. sample_submission.csv [제출양식]
>id : 평가 샘플 고유 id  
>label : 예측한 도배 하자 Class
---
### ✅ 개발환경
>tensorflow==2.16.1<br>
>click==8.1.7<br>
>pandas==2.2.2<br>
>opencv-python==4.9.0.80<br>
>imbalanced-learn==0.12.2<br>
>torch<br>
>torchvision<br>
>torchaudio<br>
---
### ✅ 모델 선정 및 학습

---
### ✅ 결과
리더보드(PUBLIC) : 275/1152(등)
![image](https://github.com/2shin0/Papering-Flaw/assets/150658909/2f9504f0-a843-4bbf-a93a-31eae3dcc79c)

---

꼭 정리하고 넘어갈 내용(base line code가 돌아가지 않았던 이유 정리)

1. opencv를 따로 설치하지 말고 자동으로 설치될 수 있도록 다른 라이브러리 설치한 경험
2. 디렉토리 주소에서 \\를 /로 변경해야 했던 경험
3. 한글 주소를 읽기 위해 암호화 후 복호화했던 경험
4. int 형을 long 형식으로 변환해줘야 했던 경험

Efficientnet4 선택 이유

데이터 처리 따로 안한 이유
