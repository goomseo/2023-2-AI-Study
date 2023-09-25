# GDSC 기초 인공지능 스터디 WIL - Week 2

(원문 URL: [GDSC 기초 인공지능 스터디 2주차 WIL - Notion](https://goomseo.notion.site/Week2-fb9fbc6ffd8d4a569ad2e1eda99643cc?pvs=4))

# 목차

1. Computer Vision Applications
2. Introduction to Pytorch
3. Pytorch Basics
4. AutoGrad & Optimizer
5. Pytorch Dataset
6. Model Save

# Computer Vision Applications

- Segmentation
    - Semantic Segmentation
        - 이미지를 픽셀 별로 분류
        - CNN vs FCN
            - CNN은 데이터의 공간적 정보를 보존 → Upsampling 시 입력과 같은 크기의 출력 생성 가능
                - Upsampling - Deconvolution / Interpolation
    - Image Segmentation
- Object Detection
    - Bounding Box로 Instance의 위치를 찾는 작업
    - R-CNN
    - SPPNet
    - Fast R-CNN
    - Faster R-CNN
    - YOLO : You Only Look Once

---

# Introduction to Pytorch

- Torch와 Caffe2를 기반으로 한 딥러닝 라이브러리
- Pytorch vs Tensorflow : Dynamic Graph vs Static Graph
    - 동적 그래프는 연산을 실행하는 동안 그래프가 생성/수정 될 수 있으며, 정적 그래프는 모든 연산이 미리 정의되어있고 그 연산들의 그래프가 구축되어있음.
- JAX
    - 구글이 개발하고 유지관리하며 사용하는 Python과 Numpy가 결합한 DL Framework이다.
    - Numpy를 **GPU**에서 연산시킬 수 있게 하여 기존 Numpy의 성능을 뛰어넘는다.
    - Framework: 응용 프로그램을 개발하기 위한 여러 라이브러리나 모듈 등을 효율적으로 사용할 수 있도록 하나로 묶어 놓은 패키지
    
    ---
    
    # Pytorch Basics
    
    - Tensor
        - 다차원 행렬을 가지는 배열(스칼라, 벡터, 행렬, 행렬 수열)로, Numpy에서의 ndarray와 같은 개념이다.
        - list나 ndarray를 이용하여 tensor를 생성할 수 있다.
    - Numpy-like Operations - Numpy를 숙지하고 있으므로 간단하게 서술
        - indexing & slicing
        - flatten
        - ones-like: 1로 채워진 Tensor를 반환
        - numpy
        - shape
        - dtype
    - Tensor Handling
        - view vs reshape : shallow copy : deep copy
        - squeeze ↔ unsqueeze
        - mm: Matrix Multiplication
        - matmul: Matrix product of two tensors (w/ broadcast)
        - broadcasting: 특정 조건 하에서 연산에 사용되는 array 중 더 작은 array가 큰 array로 broadcast되어 계산이 가능한 shape가 되게 한다.
        - nn.functional (import torch.nn.functional as F)
            - softmax: 0에서 1 사이의 값으로 모든 값을 정규화하며, 출력 값의 총합은 1.
            - one-hot encoding: 조건에 해당하는 데이터만 1, 나머지는 0으로 변환
    - AutoGrad
        - torch.autograd는 신경망 학습을 지원하는 자동 미분 패키지로, nn이 역전파 할 때 필요한 gradient 계산들(편미분, 체인 룰 등)을 대신 해준다.
        - backward: 역전파(backward propagation)

---

# AutoGrad & Optimizer

- [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
    - 딥러닝을 구성하는 레이어에 기초가 되는 클래스
- nn.Parameter
    - Tensor 객체 중 parameter의 성격을 띄는 Tensor
    - nn.Module 내에서 Autogradient의 학습 대상이 되는 Tensor
- Backward
    - 레이어에 있는 Parameter들의 미분을 수행함.
    - Forward의 결과값과 실제값의 차이를 현재 가중치에 대해 편미분 → 현재 가중치에서 이 값을 뺀 값으로 Parameter를 업데이트 함.

---

# Pytorch Dataset

- Dataset 클래스
    - 데이터 입력 형태를 정의하는 클래스로, 데이터를 입력하는 방식을 표준화한다.
    - 형식
        
        ```python
        from torch.utils.data import Dataset
        
        class CustomDataset(Dataset):
            def __init__(self,):
                pass
        
            def __len__(self):
                pass
            
            def __getitem__(self, idx):
                pass
        ```
        
- DataLoader 클래스
    - 모델 학습을 위해 데이터를 Mini Batch 단위로 제공해주며, Batch 처리가 주요 작업이다.
    - 주로 batch_size, num_workers, shuffle, drop_last 등의 attribute를 사용한다.
    - 정의
        
        ```python
        DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
                   batch_sampler=None, num_workers=0, collate_fn=None,
                   pin_memory=False, drop_last=False, timeout=0,
                   worker_init_fn=None)
        ```
        

---

# Model Save

- DL 모델은 학습 시간이 긴 편 → 학습이 정상적으로 종료되게 해야 함.
- model.state_dict(): 각 레이어를 Parameter Tensor로 mapping하는 Dictionary Object이다.

---

# Transfer Learning

- CNN 모델을 학습시키는 방식으로, Pre-trained Model을 이용하여 자신의 Data Set을 학습시킨다.
- 보유한 데이터가 적을 때 사용하면 매우 용이하다.