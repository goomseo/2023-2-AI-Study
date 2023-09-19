# GDSC 기초 인공지능 스터디 WIL - Week 1

(원문 URL: [GDSC 기초 인공지능 스터디 1주차 WIL - Notion](https://goomseo.notion.site/Week1-ee8c5dbcbc2043d8a0a43b1d9e51ea7b?pvs=4))

# 목차

1. [AI, ML, DL의 차이와 관계](https://www.notion.so/Week1-ee8c5dbcbc2043d8a0a43b1d9e51ea7b?pvs=21)
2. [DL의 구성 요소](https://www.notion.so/Week1-ee8c5dbcbc2043d8a0a43b1d9e51ea7b?pvs=21)
3. [Neural Network](https://www.notion.so/Week1-ee8c5dbcbc2043d8a0a43b1d9e51ea7b?pvs=21)
4. Nonlinear Function
5. Multi-Layer Perceptron
6. Generalization
7. Convolution Neural Network
8. 1x1 convolution
9. Modern CNN

# AI, ML, DL의 차이와 관계

AI, ML, DL은 각각 Artificial Intelligence, Machine Learning, Deep Learning의 약자로, 한국어로 하면 인공지능(혹은 AI), 기계학습(혹은 머신러닝), 심층학습(혹은 딥러닝) 정도로 표현할 수 있겠다.

우선 각각 무엇을 의미하는지 알아보자.

- AI
  - 컴퓨터과학의 세부 분야 중 하나로, 학습능력, 추론능력, 지각능력, 언어이해능력 등 인간이 가진 지적 능력을 컴퓨터를 통해 인공적으로 구현하고자 하는 분야이다.
- ML
  - AI의 한 분야로, 컴퓨터가 스스로 학습하여 인공지능의 성능을 향상시키는 기술 방법. 즉, 애플리케이션을 수정하지 않고도 데이터를 기반으로 패턴을 학습하고 결과를 예측하는 알고리즘 기법.
  - 학습시키는 방법에 따른 ML의 분류
    - Supervised Learning (지도학습): 데이터와 라벨(정답)을 함께 제공하여 학습 방법.
      - Classification (분류): 주어진 데이터를 정해진 라벨에 따라 분류하는 방식
      - Regression (회귀): 연속적인 값들을 학습하고 그에 대한 함수를 추론하여, 앞으로의 값을 예측하는 방식
    - Unsupervised Learning (비지도학습): 라벨 없이 데이터만 제공되어 이루어지는 학습 방법.
      - Clustering (군집화): 입력 받은 데이터들의 특징을 파악해 여러 개의 그룹으로 나누는 방식.
      - Dimensionality Reduction, Hidden Markov Model, etc.
    - Reinforcement Learning (강화학습): 현재 상태에서의 최적의 행위를 선택하고, 그에 대한 보상이나 벌점을 부여하며 이루어지는 학습 방법.
  - Feature
    - ML은 데이터를 분류하거나, 데이터를 통하여 앞으로의 값을 예측한다. 이 때 데이터의 값을 잘 예측하기 위한 데이터의 특징들을 Feature라고 부른다.
- DL (자세한 정보 - [클릭](https://aws.amazon.com/ko/what-is/deep-learning/))
  - ML 기술의 종류 중 하나인 인공신경망의 방법론 중 하나로, 컴퓨터가 여러 데이터를 이용하여 사람과 같이 스스로 학습할 수 있게 한다.
  - 인공신경망
    - 인간의 뉴런 구조를 본떠 만든 ML 모델로, AI를 구현하기 위한 ML의 방식 중 일종.
    - 데이터의 입력과 출력 사이에 많은 Hidden Layer를 두어, 연속된 층의 인공 신경 세포(노드)를 통해 패턴을 발견하여 스스로 학습하게 하는 것.
- ML vs DL
  ![(이미지 출처 - [클릭](https://serokell.io/blog/ai-ml-dl-difference))](https://serokell.io/files/sq/sqhrsfzw.Machine-learning-vs-deep-learning.jpg)
  (이미지 출처 - [클릭](https://serokell.io/blog/ai-ml-dl-difference))

위에 서술한 내용을 살펴보면, AI의 한 분야로 ML이 존재하고, ML 기술의 종류 중 하나인 인공신경망의 방법론 중 하나가 DL이라는 것을 알 수 있다. 즉 AI ⊃ ML ⊃ DL이다. 이를 이미지로 표현하면 아래와 같다.

![(이미지 출처 - [클릭](https://serokell.io/blog/ai-ml-dl-difference))](https://serokell.io/files/zx/zxwju3ha.Machine-learning-vs-deep-learning.jpg)

(이미지 출처 - [클릭](https://serokell.io/blog/ai-ml-dl-difference))

---

# DL의 구성 요소

- Data
  - DL에 있어 데이터는 재료라고 할 수 있다. 데이터를 이용하여 컴퓨터는 스스로 학습할 수 있다.
- Model
  ![(이미지 출처 - [클릭](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/Debugging-Deep-Learning-Model_2.png?ssl=1))](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/Debugging-Deep-Learning-Model_2.png?ssl=1)
  (이미지 출처 - [클릭](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/Debugging-Deep-Learning-Model_2.png?ssl=1))
  - 데이터의 입력으로부터 Feature를 추출하고, 우리가 원하는 출력을 만드는 프로그램이다.
  - AlexNet, GoogLeNet, DenseNet, LSTM, AutoEncoder, GAN, etc.
  - 딥러닝 모델의 종류
    - ANN (Artificial Neural Network), RNN (Recurrent Neural Network), CNN (Convolution Neural Network)
- Loss Function (손실 함수)
  - 지도학습이 이루어질 때, 알고리즘이 예측한 답과 실제 정답의 차이를 비교하기 위한 함수이다. 즉, 학습 중 알고리즘이 얼마만큼 잘못 예측하나에 대한 지표로 작용할 수 있다. 최적화를 위해 최소화하는 것이 목적인 함수로, 목적 함수 (Objective Function)이라고도 불린다.
  - 어떤 학습 방식을 이용하는 지에 따라 손실 함수는 달라진다.
    - Classification Task
      - Mean Square Error (MSE, 평균 제곱 오차):
    - Regression Task
      - Cross Entropy
    - Probabilistic Task
      - Maximum Likelihood Estimation (MLE)
- Optimization and Regularization
  - Optimization (최적화)
    - 학습 모델의 손실 함수의 값을 최소화하는 파라미터를 구하는 과정.
    - Gradient Descent Method (경사하강법)
      - 가중치의 값이 가장 작은 지점으로 손실 함수의 경사를 타고 하강하는 최적화 기법.
      - ex) SGD, Momentum, Adam, AdamW, etc.
  - [Regularization](https://pozalabs.github.io/Regularization/) (정규화)
    - 딥러닝 모델은 데이터로부터 직접 Feature를 만들어낸다. 이 때 처음부터 모델의 구조를 단순하게 설계하면 높은 수준의 Feature를 모델이 학습할 수 없다. 이를 방지하기 위해 모델의 구조를 복잡하게 설계하면 모델이 **학습 데이터를 과하게 학습(Overfitting)**하여, 학습 데이터(일반적으로, 실제 데이터의 부분 집합)에 대해서는 오차가 감소하지만 실제 데이터에 대해서는 오차가 증가하게 된다.
    - **이러한 Overfitting을 해소하기 위한 방법 중 하나가 Regularization이다.**
    - ex) Dropout, Early Stopping, Batch Normalization, MixUp, etc.

---

# Neural Network

![(이미지 출처 - [클릭](https://www.ibm.com/topics/neural-networks))](https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/cdp/cf/ul/g/3a/b8/ICLH_Diagram_Batch_01_03-DeepNeuralNetwork.png)

(이미지 출처 - [클릭](https://www.ibm.com/topics/neural-networks))

- Neural Networks, 즉 인공신경망은 Artificial Neural Networks(ANNs) 혹은 Simulated Neural Networks(SNNs)로도 알려져 있으며, DL 알고리즘의 핵심이다.
- 인간의 뉴런 구조를 본떠 만든 ML 모델로, AI를 구현하기 위한 기술 중 하나이다.
- 데이터의 입력과 출력 사이에 많은 Hidden Layer를 두어, 연속된 층의 인공 신경 세포(노드)를 통해 패턴을 발견하여 스스로 학습하게 한다.
- Neural networks are Function Approximators that stack affine transformations followed by nonlinear transformations.

---

# Nonlinear Function

- Activation Function(활성함수)로써 비선형 함수를 사용
  - 만약 활성함수가 선형함수라면, 아무리 layer를 많이 쌓아도 하나의 layer로 대체가 가능하다. DL의 목적을 이루기 위해서는 여러 겹의 hidden layers가 필요하기 때문에, layer를 쌓기 위해서는 비선형 함수를 활성함수로 사용하여야 한다.
  - 즉, 신경망의 표현성을 높이기 위해 비선형 함수를 사용한다.
  - ex) ReLU, Sigmoid, tanh, etc.

---

# Multi-Layer Perceptron

- Perceptron(퍼셉트론): 인공신경망의 한 종류로, 뉴런의 수학적 모델을 일컫는 용어이다.

---

# Generalization

- 일반화

---

# Convolution Neural Network

- d

---

# 1x1 convolution

- d

---

# Modern CNN

- d
