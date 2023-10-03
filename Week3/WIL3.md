# GDSC 기초 인공지능 스터디 WIL - Week 3

(원문 URL: [GDSC 기초 인공지능 스터디 3주차 WIL - Notion](https://goomseo.notion.site/Week3-8f862d92cbec432baef65d3e72d9a48b?pvs=4))

# 목차

1. Computer Vision
2. Image Classification
3. KNN (K-Nearest Neighbor)
4. Linear Classification
5. Multiclass SVM Loss

# Computer Vision

- Computer Vision (Inverse Rendering): 컴퓨터가 **시각적 데이터**로부터 정보를 추출하고 해석하여 특정 작업을 수행하는 인공지능의 한 분야. ↔ Computer Graphics
- Deep Learning과 CNN이 필수적인 기술이다.

---

# Image Classification

- 시각적 데이터는 Classifier를 통해 분류된다.
  - Classifier
- 인간은 시각적 데이터를 그래픽으로 받아들이지만, 컴퓨터는 숫자들의 행렬로 받아들인다.

---

# KNN (K-Nearest Neighbor)

- Simple Image Classifier
  - 훈련 단계: 아무것도 하지 않고, 모든 훈련 데이터를 기억함.
  - 예측 단계: 새로운 이미지를 가져와 앞서 훈련단계에서 보았던 데이터 중 가장 유사한 이미지를 찾음 → 그 이미지의 label로 예측
  - 각 데이터 포인트가 k개의 이웃 포인트와 비교([참고](http://vision.stanford.edu/teaching/cs231n-demos/knn/)) & Majority Vote로 카테고리를 나눈다.
  - 거리 측정 방법: L1 vs L2 → 보통 L2가 L1보다 유용하게 사용된다. (Euclidean Distance)
    - 사실 둘 다 이미지 유사도 측정에 있어서 좋은 방식이라고 할 수는 없다.
- KNN의 단점
  - 차원의 저주 (Curse of Dimensionality): 차원이 커질수록 필요한 데이터의 수가 기하급수적으로 많아진다. 1차원에서 k개의 데이터만 필요했다면, n차원에서는 k^n개의 데이터가 필요하다.
  - 시간이 오래 걸린다.
  - 픽셀 간의 거리를 측정하는 방식이 좋다고 할 수는 없다. → 원본으로부터 서로 다른 변형된 이미지들이 원본 이미지로부터 동일한 유사성 거리에 있다는 결과가 나오기도 한다.
  - Semantic Gap으로 인해 이미지의 각도, 포즈, 위치 등에 따라 이미지 분류에 어려움이 있을 수 있다.

---

# Linear Classification

- Linear Classification([참고 영상 - UMich EECS498 강의](https://www.youtube.com/watch?v=qcSEP17uKKY)): Linear Model을 이용하여 데이터들을 클래스들로 Classification하는 머신러닝 기법. 클래스 분류의 기준이 되는 Discriminant Function이 선형이다.
- KNN과 달리 가중치를 활용한다. → $f(x, W) = Wx + b$
- Image Captioning
- Parametric Approach: Linear Classification (CIFAR-10 Dataset)

---

# Multiclass SVM Loss

- $L_i = \displaystyle\sum_{j \neq y_i}max(0, s_j - s_{y_i}+\Delta)$
  - $x_i$ is the image, and $y_i$ is the label
  - $s = f(x_i, W)$ → $s_j$는 이미지에 대한 클래스의 예측 점수, $s_{y_i}$는 정답(레이블) 클래스의 예측 점수
  - $\Delta$ is a Safety Margin
- 그래프가 경첩처럼 생겨서 Hinge Loss라고도 한다.
