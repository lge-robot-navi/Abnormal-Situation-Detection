# Surveillance map anomaly trainer 

### Abstract
이 SW는 멀티 감시에이전트로부터 생성된 다층 환경지도 입력에 대해 (1)확률지도를 생성하고, (2)이상후보를 추출하고 필터링하는 딥러닝 학습 프로그램이다. 주어진 기간에 대한 다층 환경지도에 대해 통계적 데이터를 분석하여 해당 사건이 발생할 확률을 계산하여 격자지도 형태의 관측지도와 확률지도로 변환작업을 수행한다. 이후 Auto-encoder 및 CNN 기반의 딥러닝 네트워크를 사용하여 이상후보를 추출하고, 필터링한다.

This software is a program that (1) generates a probability map, (2) extracts and filters abnormal candidates, and (3) visualizes judgment information to users for multi-level environment map inputs generated from multi-monitoring agents. By analyzing statistical data for a multi-layered environment map for a given period, the probability of occurrence of the event is calculated, and the conversion is performed into an observation map and a probability map in the form of a grid map. Afterwards, anomalous candidates are extracted and filtered using an auto-encoder and a deep learning network based on CNN. 

### 

### Requirements
* Python 2.7.12 
* Tensorflow 2.0.0

# 참여기관
* ![](https://www.etri.re.kr/images/kor/sub5/signature08.png)
* [✉️](mailto:creatrix@etri.re.kr) __신호철__ ()
