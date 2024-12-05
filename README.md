# Study-on-Lightweight-Disaster-Segmentation-Model-with-Knowledge-Distillation


이 프로젝트는 [지식 증류를 이용한 재난 분할 모델 경량화 연구](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11705538)에서 제안된 방법론 및 실험을 재현하기 위해 제공됩니다.

## abstract

최근 재난 촬영 영상에 딥러닝 기반 분할(Segmentation) 모델을 사용하여 재난을 탐지하는 연구가 진행되고 있다. 재난을 빠르게 파악하고 대응해야 하는 특수성 때문에 분할 모델을 이용한 재난 상황 파악은 실시간 분석을 위해 엣지 컴퓨팅 시스템이 활용된다. 그러나 기존 분할 모델은 모델 크기와 연산량이 높아 컴퓨팅 자원이 제한된 엣지 컴퓨팅 환경에서 사용하기 어렵다. 이를 개선하기 위해, 본 논문에서는 지식 증류 기법 중 SKDS(Structured Knowledge Distillation for Semantic segmentation) 기법을 이용하여 재난 분할 모델 경량화를 수행하였다. 이를 위해 학습 시 사용되는 손실 함수 및 학습 파라미터를 조정하였다. 데이터 셋은 14개 class를 갖는 재난 이미지를 사용했으며, SKDS 기법을 이용하여 경량화를 수행한 FANet은 경량화를 수행하지 않은 기존 모델 대비 정확도 손실 없이 모델 파라미터가 45.0%, FLOPs는 20.4% 감소하였다. 

## Installation:

#### Environment:
1. Linux
2. Python 3.7 
3. Pytorch 1.8
4. NVIDIA GPU + CUDA 10.2 

#### Prepare Data
데이터 셋은 [LPCVC 2023](https://lpcv.ai/competitions/2023) 에서 제공한 데이터셋을 사용하였습니다.


#### Modify Codes
Modify `*.yml` files in `./config`
* `data:path`: path to dataset 
* `training:batch_size`: batch_size``
* `training:train_augmentations:rcrop`: input size for training
``

#### Train

```bash
train
python train.py --config ./configs/*.yml
``
distillation
python train_distillation.py --config ./configs/*.yml

pyz
python3.6 -m zipapp  solution  -p='/usr/bin/env python3.6'

```


#### test
test는 [LPCVC evaluation](https://github.com/lpcvai/23LPCVC_Segmentation_Track-Sample_Solution)을 통해 진행 하였습니다.

## reference
이 프로젝트에서 참고한 Git 관련 정보와 기타 유용한 리소스는 다음과 같습니다:
1. [23LPCV](https://lpcv.ai/competitions/2023) 2023 Low PowerComputer Vision Challenge
2. [23LPCV evaluation](https://github.com/lpcvai/23LPCVC_Segmentation_Track-Sample_Solution)  평가 방법
3. [structure_knowledge_distillation](https://github.com/irfanICMLL/structure_knowledge_distillation) sementic segmentation knowledge distillation 참조.
