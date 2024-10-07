# OWL-ViT on SageMaker

## 소개

OWL-ViT (Vision Transformer for Open-World Localization)는 Matthias Minderer, Alexey Gritsenko 등이 "[Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)" 논문에서 제안한 혁신적인 모델입니다.

### OWL-ViT란?

OWL-ViT는 다양한 (이미지, 텍스트) 쌍으로 학습된 개방형 어휘 객체 감지 네트워크입니다. 이 모델의 주요 특징은 다음과 같습니다:

- 📝 **텍스트 쿼리 기반 객체 검출**: 하나 또는 여러 개의 텍스트 쿼리를 사용하여 이미지 내에서 해당 텍스트로 설명된 대상 객체를 검색하고 감지할 수 있습니다.
- 🌐 **개방형 어휘**: 사전 정의된 객체 클래스에 제한되지 않고, 다양한 객체를 인식할 수 있습니다.
- 🖼️ **이미지-텍스트 쌍 학습**: 다양한 이미지와 텍스트 쌍을 통해 학습되어, 풍부한 시각적 이해와 언어적 연관성을 갖추고 있습니다.

> **주목할 점**: OWL-ViT는 **85M(patch 32 : 492 MB) ~ 87M(patch 16 : 611 MB) 파라미터**로 구성된 비교적 작은 크기의 모델입니다. 이러한 컴팩트한 구조 덕분에 **CPU에서도 효율적인 추론이 가능**합니다.

이러한 특성은 다음과 같은 이점을 제공합니다:

- 🚀 **빠른 추론 속도**: 작은 모델 크기로 인해 추론 시간이 단축됩니다.
- 💰 **비용 효율성**: CPU 인스턴스를 사용할 수 있어 GPU 대비 비용을 절감할 수 있습니다.
- 🔧 **유연한 배포**: 다양한 환경에서 쉽게 배포할 수 있습니다.
- 🎯 **다목적 사용**: 객체 검출, 이미지 검색, 시각적 질의응답 등 다양한 작업에 활용 가능합니다.

이 repository는 이러한 강력한 OWL-ViT 모델을 Amazon SageMaker에서 다양한 방식으로 추론하는 방법을 제공합니다.

## 주요 노트북 소개

이 repository에서는 다음과 같은 네 가지 주요 추론 방식을 다룹니다:

1. **SageMaker Processing Job** [1.SageMaker-Processing-Job.ipynb](1.SageMaker-Processing-Job.ipynb)
2. **SageMaker BatchTransform Job** [2.SageMaker-BatchTransform-Job.ipynb](2.SageMaker-BatchTransform-Job.ipynb)
3. **SageMaker Real-time Inference** [3.SageMaker-Realtime-Inference.ipynb](3.SageMaker-Realtime-Inference.ipynb) 
4. **SageMaker Serverless Inference** [4.SageMaker-Serverless-Inference.ipynb](4.SageMaker-Serverless-Inference.ipynb) 

각 방식에 대한 자세한 설명은 아래와 같습니다.

### 1. SageMaker Processing Job

```python
current_time = strftime("%m%d-%H%M%s")
i_type = instance_type.replace('.','-')
job_name = f'owl-vit-{i_type}-{instance_count}-{current_time}'

eval_processor = FrameworkProcessor(
    PyTorch,
    framework_version="2.3",
    py_version="py311",
    role=role, 
    instance_count=instance_count,
    instance_type=instance_type,
    sagemaker_session=sagemaker_session
    )

eval_processor.run(
    code="evaluation.py",
    source_dir=source_dir,
    wait=False,
    inputs=[ProcessingInput(source=input_image_path, 
                            input_name="test_data", 
                            destination="/opt/ml/processing/data", 
                            s3_data_distribution_type=s3_data_distribution_type),
            ProcessingInput(source=model_weight_path, 
                            input_name="model_weight", 
                            destination="/opt/ml/processing/weights")
    ],
    outputs=[
        ProcessingOutput(source="/opt/ml/processing/output", destination=output_path),
    ],
    arguments=["--threshold", "0.1"],
    job_name=job_name
)
```

- 대량의 데이터를 배치 방식으로 처리
- 결과를 CSV 파일로 S3에 저장
- 대규모 데이터셋에 적합

### 2. SageMaker BatchTransform Job

```python
create_model_response = sm_client.create_model(
    ModelName=sm_model_name, 
    ExecutionRoleArn=role, 
    PrimaryContainer=container,
)

import json
texts = json.dumps([["a photo of a tv", "a photo of a dog"]])

env = {"threshold" : "0.1",
       "texts" : texts}

response = sm_client.create_transform_job(
    TransformJobName=job_name,
    ModelName=sm_model_name,
    MaxConcurrentTransforms=2,
    MaxPayloadInMB=2,
    BatchStrategy="SingleRecord", ##'MultiRecord',
    Environment=env,
    TransformInput={
        'DataSource': {
            'S3DataSource': {
                'S3DataType': 'S3Prefix',
                'S3Uri': input_image_path 
            }
        },
        'ContentType': "application/x-image",
    },
    TransformOutput={
        'S3OutputPath': output_path,
        'Accept': 'application/json',
    },
    TransformResources={
        'InstanceType': 'ml.m5.2xlarge',
        'InstanceCount': 1
    }
)
```

- 배치 방식으로 대량 데이터 처리
- 각 입력에 대한 결과를 개별 .out 파일로 S3에 저장
- 개별 결과 추적이 필요한 경우에 유용

### 3. SageMaker Real-time Inference

```python
create_model_response = sm_client.create_model(
    ModelName=sm_model_name, 
    ExecutionRoleArn=role, 
    PrimaryContainer=container,
)

create_endpoint_config_response = sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "InstanceType": instance_type,
            "InitialVariantWeight": 1,
            "InitialInstanceCount": 1,
            "ModelName": sm_model_name,
            "VariantName": "AllTraffic",
        }
    ],
)
create_endpoint_response = sm_client.create_endpoint(
    EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
)
```

- 실시간 추론을 위한 지속적인 엔드포인트 제공
- 낮은 지연 시간이 요구되는 애플리케이션에 적합
- 24/7 가용성 필요 시 사용

### 4. SageMaker Serverless Inference

```python
create_model_response = sm_client.create_model(
    ModelName=sm_model_name, ExecutionRoleArn=role, PrimaryContainer=container
)

create_endpoint_config_response = sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "ModelName": sm_model_name,
            "VariantName": "AllTraffic",
            "ServerlessConfig": {
                "MemorySizeInMB": 2048,
                "MaxConcurrency": 2,
                "ProvisionedConcurrency": 1,
            }
        }
    ],
)
create_endpoint_response = sm_client.create_endpoint(
    EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
)
```

- 요청 시에만 서버가 동작하는 서버리스 방식
- 비용 효율적이며 자동 스케일링 지원
- 간헐적인 트래픽 패턴에 적합
- 첫 호출 시 Cold Start 발생 가능

## 설치 및 사용법

### 사용법

1. 이 저장소를 클론합니다:
```bash
git clone https://github.com/your-username/OWL-ViT-on-SageMaker.git
cd OWL-ViT-on-SageMaker
```

2. 각 노트북 파일을 열어 단계별 가이드를 따라 실행하세요.


## 라이선스

이 프로젝트는 Apache-2.0 라이선스 하에 배포됩니다.


# OWL-ViT on SageMaker

## Introduction

OWL-ViT (Vision Transformer for Open-World Localization) is an innovative model proposed by Matthias Minderer, Alexey Gritsenko, and others in their paper "[Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)".

### What is OWL-ViT?

OWL-ViT is an open-vocabulary object detection network trained on various (image, text) pairs. The key features of this model include:

- 📝 **Text Query-Based Object Detection**: It can search and detect target objects described in text within an image using one or multiple text queries.
- 🌐 **Open Vocabulary**: It can recognize a wide range of objects without being limited to predefined object classes.
- 🖼️ **Image-Text Pair Learning**: Trained on diverse image and text pairs, it possesses rich visual understanding and linguistic relevance.

> **Note**: OWL-ViT is a relatively small model with **85M (patch 32: 492 MB) ~ 87M (patch 16: 611 MB) parameters**. This compact structure enables **efficient inference even on CPUs**.

These characteristics offer the following advantages:

- 🚀 **Fast Inference Speed**: The small model size reduces inference time.
- 💰 **Cost-Effectiveness**: CPU instances can be used, reducing costs compared to GPUs.
- 🔧 **Flexible Deployment**: Easy to deploy in various environments.
- 🎯 **Multipurpose Use**: Applicable to various tasks such as object detection, image search, and visual question answering.

This repository provides methods for inferencing this powerful OWL-ViT model on Amazon SageMaker in various ways.

## Key Notebooks Introduction

This repository covers four main inference methods:

1. **SageMaker Processing Job** [1.SageMaker-Processing-Job.ipynb](1.SageMaker-Processing-Job.ipynb)
2. **SageMaker BatchTransform Job** [2.SageMaker-BatchTransform-Job.ipynb](2.SageMaker-BatchTransform-Job.ipynb)
3. **SageMaker Real-time Inference** [3.SageMaker-Realtime-Inference.ipynb](3.SageMaker-Realtime-Inference.ipynb) 
4. **SageMaker Serverless Inference** [4.SageMaker-Serverless-Inference.ipynb](4.SageMaker-Serverless-Inference.ipynb) 

Detailed explanations for each method are as follows:

### 1. SageMaker Processing Job

```python
current_time = strftime("%m%d-%H%M%s")
i_type = instance_type.replace('.','-')
job_name = f'owl-vit-{i_type}-{instance_count}-{current_time}'

eval_processor = FrameworkProcessor(
    PyTorch,
    framework_version="2.3",
    py_version="py311",
    role=role, 
    instance_count=instance_count,
    instance_type=instance_type,
    sagemaker_session=sagemaker_session
    )

eval_processor.run(
    code="evaluation.py",
    source_dir=source_dir,
    wait=False,
    inputs=[ProcessingInput(source=input_image_path, 
                            input_name="test_data", 
                            destination="/opt/ml/processing/data", 
                            s3_data_distribution_type=s3_data_distribution_type),
            ProcessingInput(source=model_weight_path, 
                            input_name="model_weight", 
                            destination="/opt/ml/processing/weights")
    ],
    outputs=[
        ProcessingOutput(source="/opt/ml/processing/output", destination=output_path),
    ],
    arguments=["--threshold", "0.1"],
    job_name=job_name
)
```

- Processes large amounts of data in batch mode
- Saves results as CSV files in S3
- Suitable for large-scale datasets

### 2. SageMaker BatchTransform Job

```python
create_model_response = sm_client.create_model(
    ModelName=sm_model_name, 
    ExecutionRoleArn=role, 
    PrimaryContainer=container,
)

import json
texts = json.dumps([["a photo of a tv", "a photo of a dog"]])

env = {"threshold" : "0.1",
       "texts" : texts}

response = sm_client.create_transform_job(
    TransformJobName=job_name,
    ModelName=sm_model_name,
    MaxConcurrentTransforms=2,
    MaxPayloadInMB=2,
    BatchStrategy="SingleRecord", ##'MultiRecord',
    Environment=env,
    TransformInput={
        'DataSource': {
            'S3DataSource': {
                'S3DataType': 'S3Prefix',
                'S3Uri': input_image_path 
            }
        },
        'ContentType': "application/x-image",
    },
    TransformOutput={
        'S3OutputPath': output_path,
        'Accept': 'application/json',
    },
    TransformResources={
        'InstanceType': 'ml.m5.2xlarge',
        'InstanceCount': 1
    }
)
```

- Processes large amounts of data in batch mode
- Saves results as individual .out files in S3 for each input
- Useful when individual result tracking is needed

### 3. SageMaker Real-time Inference

```python
create_model_response = sm_client.create_model(
    ModelName=sm_model_name, 
    ExecutionRoleArn=role, 
    PrimaryContainer=container,
)

create_endpoint_config_response = sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "InstanceType": instance_type,
            "InitialVariantWeight": 1,
            "InitialInstanceCount": 1,
            "ModelName": sm_model_name,
            "VariantName": "AllTraffic",
        }
    ],
)
create_endpoint_response = sm_client.create_endpoint(
    EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
)
```

- Provides continuous endpoints for real-time inference
- Suitable for applications requiring low latency
- Used when 24/7 availability is needed

### 4. SageMaker Serverless Inference

```python
create_model_response = sm_client.create_model(
    ModelName=sm_model_name, ExecutionRoleArn=role, PrimaryContainer=container
)

create_endpoint_config_response = sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "ModelName": sm_model_name,
            "VariantName": "AllTraffic",
            "ServerlessConfig": {
                "MemorySizeInMB": 2048,
                "MaxConcurrency": 2,
                "ProvisionedConcurrency": 1,
            }
        }
    ],
)
create_endpoint_response = sm_client.create_endpoint(
    EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
)
```

- Serverless approach where the server operates only upon request
- Cost-effective and supports automatic scaling
- Suitable for intermittent traffic patterns
- Cold start may occur on the first call

## Installation and Usage

### Usage

1. Clone this repository:
```bash
git clone https://github.com/your-username/OWL-ViT-on-SageMaker.git
cd OWL-ViT-on-SageMaker
```

2. Open each notebook file and follow the step-by-step guide.

## License

This project is distributed under the Apache-2.0 license.