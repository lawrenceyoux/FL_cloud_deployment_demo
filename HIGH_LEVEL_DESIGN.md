# Federated Learning Demo: Healthcare Risk Prediction
## High-Level Design Document

> **Note**: This document describes the original planned architecture and has not been fully implemented.
> The current running code lives in `local_dev/` and covers:
> - Stroke prediction (binary classification) on the Kaggle stroke dataset
> - Simple `StrokeNet` MLP, FedAvg aggregation, 3 non-IID hospitals
> - Flower server/client mode OR single-process simulation via `simulate.py`
> - MLflow for experiment tracking
>
> Features described here (multimodal data, AWS ECS, differential privacy, Streamlit dashboard)
> are **not yet implemented**. See [README.md](README.md) for what currently works.

---

## 1. Project Overview

### 1.1 Problem Statement
Demonstrate federated learning for **multi-hospital stroke risk prediction** using multimodal data while preserving patient privacy. The system will perform:
- **Classification**: Predict stroke risk (binary classification with risk stratification)
- **Clustering**: Identify patient subgroups for personalized prevention strategies
- **Representation Learning**: Learn shared representations across hospitals without data sharing
- **Data Fusion**: Combine heterogeneous features (demographics, clinical, lifestyle) effectively

### 1.2 Use Case: Multi-Hospital Stroke Prevention Consortium
- **Scenario**: 3-5 hospitals want to collaboratively train ML models without sharing raw patient data
- **Real Dataset**: Stroke Prediction Dataset (Kaggle, 5,110 patients, 12 features)
- **Data Modalities**: 
  - Demographics: age, gender, marital status, residence type
  - Medical History: hypertension, heart disease
  - Lifestyle Factors: work type, smoking status
  - Clinical Measurements: average glucose level, BMI
- **Privacy Guarantee**: Differential privacy + secure aggregation
- **Challenge**: Non-IID data (Hospital A: elderly urban, Hospital B: young professionals, Hospital C: rural mixed)

---

## 2. System Architecture

### 2.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    AWS Cloud Environment (EKS)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │        FL Orchestration Layer (EKS Cluster)             │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │     Flower Server (Kubernetes Deployment)        │   │   │
│  │  │  - Model aggregation (FedAvg, FedProx, FedOpt)   │   │   │
│  │  │  - Round management & scheduling                 │   │   │
│  │  │  - Differential privacy budget tracking          │   │   │
│  │  │  - Representation aggregation                    │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│              ┌───────────────┼───────────────┐                  │
│              │               │               │                  │
│  ┌───────────▼────┐  ┌──────▼──────┐  ┌────▼──────────┐       │
│  │  Hospital 1    │  │ Hospital 2  │  │  Hospital 3   │       │
│  │  (K8s Pod)     │  │ (K8s Pod)   │  │  (K8s Pod)    │       │
│  │                │  │             │  │               │       │
│  │ ┌────────────┐ │  │ ┌─────────┐ │  │ ┌───────────┐ │       │
│  │ │Local Data  │ │  │ │Local    │ │  │ │Local Data │ │       │
│  │ │(S3 Bucket) │ │  │ │Data     │ │  │ │(S3)       │ │       │
│  │ └────────────┘ │  │ └─────────┘ │  │ └───────────┘ │       │
│  │ ┌────────────┐ │  │ ┌─────────┐ │  │ ┌───────────┐ │       │
│  │ │FL Client   │ │  │ │FL Client│ │  │ │FL Client  │ │       │
│  │ │+ Trainer   │ │  │ │+ Trainer│ │  │ │+ Trainer  │ │       │
│  │ └────────────┘ │  │ └─────────┘ │  │ └───────────┘ │       │
│  └────────────────┘  └─────────────┘  └───────────────┘       │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              MLOps & Monitoring Layer                    │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │   MLflow     │  │  CloudWatch  │  │  S3 Model    │  │   │
│  │  │  Tracking    │  │   Logs &     │  │  Registry    │  │   │
│  │  │   Server     │  │   Metrics    │  │              │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  │  ┌──────────────┐  ┌──────────────┐                    │   │
│  │  │  Streamlit   │  │   DynamoDB   │                    │   │
│  │  │  Dashboard   │  │  Metadata    │                    │   │
│  │  └──────────────┘  └──────────────┘                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Infrastructure as Code (Terraform)             │   │
│  │  - VPC, Security Groups, IAM Roles                       │   │
│  │  - ECS Cluster, Task Definitions                         │   │
│  │  - S3 Buckets, DynamoDB Tables                           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Breakdown

#### **2.2.1 Data Layer**
- **Storage**: S3 Data Lake + Delta Lake (versioned)
- **Data Format**: Parquet files for efficient columnar storage
- **Catalog**: AWS Glue Data Catalog + Athena for queries
- **Dataset**: **Stroke Prediction Dataset** (Kaggle, 5,110 patients)
  - Demographics: age, gender, marital status, residence type
  - Medical History: hypertension (binary), heart disease (binary)
  - Lifestyle: work type (categorical), smoking status (categorical)
  - Clinical: average glucose level (continuous), BMI (continuous)
  - Target: stroke (binary: 0/1, ~5% positive class - imbalanced)
- **Distribution**: Non-IID split across hospitals
  - Hospital 1 (1,700 patients): Elderly urban, 12% stroke rate
  - Hospital 2 (1,400 patients): Young professionals, 2% stroke rate
  - Hospital 3 (2,010 patients): Rural mixed population, 5% stroke rate

#### **2.2.2 Federated Learning Layer** ⭐
- **Framework**: Flower (flwr.dev) - production-ready FL framework
- **Container Orchestration**: Amazon EKS (Kubernetes)
- **Distributed Training**: Ray for parallel client execution

- **Server Component** (Kubernetes Deployment):
  - Coordinates FL rounds across distributed clients
  - **Aggregation Strategies**: FedAvg, FedProx, FedOpt, FedAdam
  - **Representation Aggregation**: Aggregate learned embeddings
  - Applies differential privacy noise (Gaussian mechanism)
  - Stores global model to S3 Model Registry
  - Client selection and scheduling
  - Handles stragglers and dropouts
  
- **Client Components** (Kubernetes Pods, one per hospital):
  - Local training on hospital data (never leaves premises)
  - **Representation Learning**: Learn local embeddings
  - **Data Fusion**: Combine heterogeneous features locally
  - Computes model updates (gradients, not raw data)
  - Sends encrypted updates to server
  - Never exposes patient-level information
  - GPU-accelerated training (optional)

- **Distributed ML Features**:
  - **Multi-GPU Training**: DataParallel within each client
  - **Gradient Compression**: Reduce communication overhead
  - **Asynchronous Updates**: Support for async FL
  - **Personalization**: Local fine-tuning after global training

#### **2.2.3 ML Pipeline** (MLflow-Tracked End-to-End)
```
Data Ingestion → Feature Engineering → Representation Learning → Federated Training → Evaluation → Deployment
     │                   │                        │                      │                │            │
  S3 + Spark       SageMaker Process      Autoencoder/Fusion      Flower + EKS         MLflow    SageMaker Endpoint
  [MLflow: log_dataset]  [MLflow: log_params]  [MLflow: log_model]  [MLflow: log_metrics]  [Model Registry]
```

**Stage-by-Stage MLflow Integration**:
1. **Data Prep**: Log dataset version, statistics, quality metrics
2. **Feature Engineering**: Log preprocessing pipeline, feature importance
3. **Baseline Training**: Log centralized model (comparison benchmark)
4. **FL Training**: Log per-round metrics, client participation, privacy budget
5. **Evaluation**: Log confusion matrix, ROC curves, fairness metrics
6. **Model Registry**: Version and stage models (staging → production)
7. **Deployment**: Track predictions, monitor drift

**Models**:
1. **Classification Model** (Stroke Prediction): 
   - **Architecture**: Neural network with representation learning
   - **Input**: 12 features (demographics, clinical, lifestyle)
   - **Representation Layer**: 64-dim learned embeddings (data fusion)
   - **Classification Head**: 3-layer MLP → Binary output (stroke yes/no)
   - **Fusion Strategy**: Early fusion (concatenate all features) + learned embeddings
   - **Handling Imbalance**: Focal loss + class weights
   - **Output**: Stroke probability + risk stratification (High/Medium/Low based on threshold)

2. **Representation Learning Component**:
   - **Autoencoder**: Learn compressed representations locally
   - **Federated**: Aggregate encoder weights only
   - **Embedding Dim**: 64 (from 12 input features)
   - **Purpose**: 
     - Capture shared patterns across hospitals
     - Enable transfer learning
     - Reduce dimensionality
     - Improve generalization
   - **Technique**: Contrastive learning + supervised pre-training

3. **Data Fusion Approaches**:
   - **Early Fusion**: Concatenate all features → Joint representation
   - **Late Fusion**: Separate encoders per modality → Combine predictions
   - **Attention Fusion**: Learn importance weights for each feature type
   - **Cross-Modal Learning**: Learn correlations between demographics, clinical, lifestyle

#### **2.2.4 Privacy & Security**
- **Differential Privacy**: 
  - Gaussian noise added to gradients (ε=1.0, δ=1e-5)
  - Per-example gradient clipping
  - Privacy accountant tracking

- **Secure Aggregation**: 
  - Homomorphic encryption for gradient aggregation
  - TLS 1.3 for all communications
  - IAM roles with least privilege

- **Compliance**:
  - HIPAA-aligned (simulated)
  - Audit logging to CloudWatch

#### **2.2.5 MLOps Layer**
- **Experiment Tracking**: MLflow hosted on EC2/ECS
  - Track hyperparameters, metrics, models
  - Compare federated vs centralized baselines
  
- **Monitoring**: CloudWatch + Custom Metrics
  - Training progress per round
  - Model accuracy, AUC-ROC, F1-score
  - System resource utilization
  - Privacy budget consumption

- **Orchestration**: Step Functions or Airflow
  - Automated training pipeline
  - Scheduled retraining
  - Model deployment workflow

- **Visualization**: Streamlit Dashboard
  - Real-time training progress
  - Model performance comparison
  - Privacy metrics
  - Client participation status

---

## 3. Data Pipeline Architecture

### 3.1 Data Generation
```python
# Synthetic data generator
- Longitudinal patient records (6-12 months)
- Realistic correlations between modalities
- Different data distributions per hospital (non-IID)
- Missing data patterns (realistic)
```

### 3.2 Data Preprocessing
```
Raw Data → Quality Checks → Normalization → Feature Engineering → Train/Val Split
```

### 3.3 Feature Engineering
- **Temporal Features**: Moving averages, trends, variance
- **Multimodal Fusion**: Early/late fusion strategies
- **Dimensionality Reduction**: PCA/UMAP for omics data
- **Embedding Layer**: Learned representations

---

## 4. Federated Learning Workflow

### 4.1 Training Process
```
Round 1:
  1. Server initializes global model
  2. Broadcast model to all hospitals
  3. Each hospital trains locally (5 epochs)
  4. Hospitals send model updates (gradients)
  5. Server aggregates updates (FedAvg)
  6. Server applies differential privacy
  7. Update global model
  
Round 2-N: Repeat

Stopping Criteria:
  - Max rounds reached (50-100)
  - Global model converges (loss plateau)
  - Privacy budget exhausted
```

### 4.2 Aggregation Strategy
- **FedAvg**: Weighted average by hospital dataset size
- **FedProx**: Handles device heterogeneity
- **Adaptive**: Learning rate scheduling

### 4.3 Evaluation
- **Local Evaluation**: Each hospital evaluates on local test set
- **Global Evaluation**: Aggregated metrics
- **Fairness Metrics**: Performance across hospitals
- **Comparison**: FL vs centralized vs local-only models

---

## 5. AWS Infrastructure

### 5.1 Compute Resources

#### **EKS Cluster Configuration**
```yaml
EKS_Cluster:
  name: fl-stroke-prediction
  version: 1.28
  region: us-east-1
  
  node_groups:
    system:
      instance_type: t3.medium
      desired_capacity: 2
      min_size: 2
      max_size: 5
      purpose: System pods (CoreDNS, metrics-server, etc.)
    
    training:
      instance_type: c5.2xlarge  # 8 vCPU, 16GB RAM
      desired_capacity: 3
      min_size: 3
      max_size: 10
      purpose: FL clients (hospital nodes)
      labels:
        workload: federated-learning
      taints:
        - key: fl-training
          value: "true"
          effect: NoSchedule
    
    gpu_training:  # Optional for large models
      instance_type: g4dn.xlarge  # 4 vCPU, 16GB RAM, 1 GPU
      desired_capacity: 0
      min_size: 0
      max_size: 5
      purpose: GPU-accelerated training
  
  add_ons:
    - vpc-cni
    - coredns
    - kube-proxy
    - aws-ebs-csi-driver  # Persistent volumes
    - aws-load-balancer-controller  # Ingress
    - cluster-autoscaler
    - metrics-server
```

#### **Kubernetes Workloads**
```yaml
FL_Server:
  kind: Deployment
  replicas: 1
  resources:
    requests:
      cpu: 2
      memory: 4Gi
    limits:
      cpu: 4
      memory: 8Gi
  storage:
    pvc: 50Gi (EBS gp3)

FL_Clients:
  kind: Deployment  # or StatefulSet for persistent identity
  replicas: 3  # One per hospital
  resources:
    requests:
      cpu: 1
      memory: 2Gi
    limits:
      cpu: 2
      memory: 4Gi

MLflow_Server:
  kind: StatefulSet
  replicas: 1
  resources:
    requests:
      cpu: 1
      memory: 2Gi
  storage:
    pvc: 100Gi (EBS gp3 for artifacts)

Streamlit_Dashboard:
  kind: Deployment
  replicas: 1
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
```

#### **Big Data Processing (Optional for Large Scale)**
```yaml
EMR_Cluster:  # For Spark-based preprocessing
  name: fl-data-processing
  release: emr-6.15.0
  applications: [Spark, Hadoop, Hive, JupyterHub]
  
  instance_groups:
    master:
      instance_type: m5.xlarge
      count: 1
    
    core:
      instance_type: m5.2xlarge
      count: 3
      auto_scaling:
        min: 3
        max: 20
        policy: YarnMemoryAvailable
  
  use_cases:
    - Large-scale data preprocessing (>10M records)
    - Feature engineering with Spark ML
    - Delta Lake for versioned data storage
    - Distributed data validation

Ray_Cluster:  # Alternative for distributed Python
  head_node:
    instance_type: m5.2xlarge
    count: 1
  
  worker_nodes:
    instance_type: c5.4xlarge
    count: 5
    auto_scaling: true
  
  use_cases:
    - Distributed hyperparameter tuning (Ray Tune)
    - Parallel FL client simulation
    - Distributed data preprocessing (Ray Data)
    - Reinforcement learning for FL optimization
```

- **Auto-scaling**: 
  - Kubernetes HPA (Horizontal Pod Autoscaler) based on CPU/memory
  - Cluster Autoscaler for node provisioning
  - Karpenter for advanced scheduling (spot instances)

### 5.2 Storage
- **S3 Buckets**:
  - `fl-demo-data-hospital-{1,2,3}`: Local datasets
  - `fl-demo-models`: Global models and checkpoints
  - `fl-demo-artifacts`: MLflow artifacts
  - `fl-demo-logs`: Application logs

- **DynamoDB**: 
  - Training metadata
  - Round status tracking
  - Client participation records

### 5.3 Networking
- **VPC**: Isolated network environment
- **Security Groups**: Restrict traffic between components
- **NAT Gateway**: Outbound internet access
- **ALB**: Load balancer for MLflow and dashboard

### 5.4 Cost Estimation (Monthly)

#### **EKS-Based Architecture**
```
EKS Control Plane: $73 (fixed)
EC2 Worker Nodes:
  - System nodes (2x t3.medium, 24/7): ~$60
  - Training nodes (3x c5.2xlarge, 8h/day): ~$80
  Subtotal: ~$140

Storage:
  - S3 Storage (10 GB): ~$0.23
  - EBS Volumes (200 GB gp3): ~$16
  - S3 Requests: ~$5
  Subtotal: ~$21

Data Lake (Optional):
  - EMR Cluster (on-demand, 10h/month): ~$50
  - Or Ray Cluster (spot instances): ~$20
  Subtotal: ~$20-50

MLOps:
  - ALB (Application Load Balancer): ~$16
  - DynamoDB (on-demand): ~$5
  - CloudWatch Logs + Metrics: ~$15
  Subtotal: ~$36

Networking:
  - NAT Gateway: ~$32
  - Data Transfer (100 GB): ~$9
  Subtotal: ~$41

SageMaker (Optional):
  - Processing Jobs (10h/month): ~$20
  - Inference Endpoint (ml.m5.large, 24/7): ~$120
  Subtotal: ~$140 (only if using SageMaker)

---------------------------------
Total (EKS + basic): ~$258/month
Total (with EMR): ~$288/month
Total (with SageMaker): ~$398/month

Cost Optimizations:
  - Use Spot Instances (save 70%): -$56
  - Shutdown off-hours (16h/day): -$60
  - Fargate Spot for batch jobs: -$20
  - Use VPC Endpoints (vs NAT): -$25
  - Reserved Instances (1-year): -$30
  
Optimized Monthly Cost: ~$120-150/month
```

#### **Comparison: EKS vs ECS**
| Feature | EKS | ECS Fargate |
|---------|-----|-------------|
| Monthly Cost | $258 (optimized: $120) | $167 |
| GPU Support | ✅ Native | ⚠️ Limited |
| Portability | ✅ Multi-cloud | ❌ AWS-only |
| ML Ecosystem | ✅ Rich (Kubeflow, KServe) | ⚠️ Basic |
| Complex Scheduling | ✅ Advanced | ⚠️ Limited |
| Learning Curve | ⚠️ Steeper | ✅ Simpler |
| Best For | Production, Scale, ML | Simple apps, Serverless |

**Recommendation**: Use **EKS** for production ML workloads with growth potential

---

## 6. Deployment Strategy

### 6.1 Infrastructure as Code
```
terraform/
├── main.tf              # Main configuration
├── vpc.tf               # Network setup
├── ecs.tf               # Container definitions
├── s3.tf                # Storage buckets
├── iam.tf               # Permissions
├── monitoring.tf        # CloudWatch
└── variables.tf         # Configuration
```

### 6.2 Container Images
```
Docker/
├── Dockerfile.server    # FL server image
├── Dockerfile.client    # FL client image
├── Dockerfile.mlflow    # MLflow tracking
└── Dockerfile.dashboard # Visualization
```

### 6.3 Deployment Steps
```bash
1. terraform init && terraform apply    # Infrastructure
2. python data_generator.py             # Generate synthetic data
3. aws s3 sync ./data s3://buckets      # Upload data
4. ./deploy_containers.sh               # Deploy ECS tasks
5. python start_training.py             # Trigger FL training
6. streamlit run dashboard.py           # Launch monitoring
```

### 6.4 CI/CD Pipeline (GitHub Actions)
```
PR → Tests → Build Images → Push to ECR → Deploy to ECS → Smoke Tests
```

---

## 7. Evaluation & Metrics

### 7.1 Model Performance
- **Classification**: 
  - Accuracy, Precision, Recall, F1-Score
  - AUC-ROC, AUC-PR
  - Confusion matrix per hospital
  
- **Clustering**:
  - Silhouette score
  - Davies-Bouldin index
  - Clinical interpretability

### 7.2 Federated Learning Metrics
- **Convergence**: Rounds to convergence
- **Communication Efficiency**: Bytes transferred
- **Fairness**: Performance variance across hospitals
- **Privacy Cost**: Privacy budget (ε) consumed

### 7.3 Comparison Baselines
- **Centralized**: All data pooled (upper bound)
- **Local-only**: Each hospital trains independently (lower bound)
- **Federated**: Our approach (privacy-utility tradeoff)

---

## 8. Research Contributions

### 8.1 Novel Aspects & Demonstrated Concepts

#### **A. Distributed and Federated Machine Learning** ✅
1. **Federated Averaging (FedAvg)**: Weighted aggregation of local models
2. **Federated Optimization**: FedProx, FedOpt, FedAdam for non-IID data
3. **Client Selection**: Strategic sampling for efficiency
4. **Asynchronous FL**: Handle stragglers and dropouts
5. **Personalized FL**: Local fine-tuning after global training
6. **Vertical FL**: Explore feature-split learning (future work)
7. **Cross-Silo FL**: Enterprise/hospital setting (vs cross-device)
8. **Distributed Training**: Multi-GPU within clients using DataParallel
9. **Communication Efficiency**: Gradient compression, quantization, sparsification
10. **Fairness**: Ensure performance equity across hospitals

**Demo Coverage**:
- ✅ Multiple FL algorithms (FedAvg, FedProx)
- ✅ Non-IID data distribution simulation
- ✅ Client dropout handling
- ✅ Aggregation visualization
- ✅ Communication cost tracking
- ✅ Fairness metrics per hospital

#### **B. Data Fusion and Representation Learning** ✅
1. **Representation Learning**:
   - **Autoencoders**: Learn compressed representations (12 → 64 dim embeddings)
   - **Contrastive Learning**: Maximize agreement between similar patients
   - **Transfer Learning**: Pre-train encoder, fine-tune classifier
   - **Federated Representations**: Aggregate encoder weights across hospitals
   - **Embedding Space Analysis**: Visualize learned representations with t-SNE/UMAP

2. **Data Fusion Techniques**:
   - **Early Fusion**: Concatenate demographics + clinical + lifestyle → Joint encoder
   - **Late Fusion**: Separate encoders → Combine predictions (ensemble)
   - **Attention-Based Fusion**: Learn importance weights for each modality
     ```python
     # Example: Attention fusion
     demographic_emb = demo_encoder(demographics)  # [batch, 16]
     clinical_emb = clinical_encoder(glucose, bmi)  # [batch, 32]
     lifestyle_emb = lifestyle_encoder(smoking, work)  # [batch, 16]
     
     # Stack embeddings
     all_embs = torch.stack([demographic_emb, clinical_emb, lifestyle_emb], dim=1)  # [batch, 3, hidden]
     
     # Attention weights
     attention_weights = softmax(W @ all_embs)  # [batch, 3, 1]
     
     # Weighted combination
     fused = (attention_weights * all_embs).sum(dim=1)  # [batch, hidden]
     ```
   - **Cross-Modal Learning**: Learn correlations (e.g., glucose ↔ BMI ↔ age)
   - **Hierarchical Fusion**: Demographics → Clinical → Lifestyle (sequential)

3. **Multimodal Challenges in FL**:
   - **Heterogeneous Features**: Different modalities have different scales
   - **Missing Modalities**: Some hospitals lack certain features
   - **Modality Alignment**: Ensure consistent encoding across hospitals
   - **Privacy Leakage**: Prevent modality-specific information leakage

**Demo Coverage**:
- ✅ Autoencoder for representation learning
- ✅ Multiple fusion strategies (early, late, attention)
- ✅ Learned embeddings visualization
- ✅ Feature importance analysis
- ✅ Cross-modal correlation heatmaps
- ✅ Federated representation aggregation
- ✅ Transfer learning evaluation

#### **C. Privacy-Utility Tradeoff**
1. Empirical analysis of differential privacy impact on accuracy
2. Privacy budget optimization (ε selection)
3. Gradient clipping effects on convergence

#### **D. Non-IID Data Handling**
1. Label skew: Different stroke rates per hospital
2. Feature skew: Different age distributions
3. Techniques: FedProx, client weighting, data augmentation

### 8.2 Publishable Outputs
- Conference paper: "Privacy-Preserving Federated Learning for Multimodal Healthcare Risk Prediction"
- Reproducible benchmark dataset
- Open-source FL framework extensions
- Best practices guide for healthcare FL deployment

---

## 9. Project Structure

```
FL/
├── README.md
├── HIGH_LEVEL_DESIGN.md (this file)
├── requirements.txt
├── setup.py
│
├── data/
│   ├── generate_synthetic_data.py
│   ├── data_splitter.py          # Split data across hospitals
│   └── data_validator.py
│
├── src/
│   ├── models/
│   │   ├── multimodal_classifier.py
│   │   ├── federated_clustering.py
│   │   └── fusion_layers.py
│   │
│   ├── federated/
│   │   ├── server.py             # Flower server
│   │   ├── client.py             # Flower client
│   │   ├── aggregation.py        # Custom aggregation
│   │   └── privacy.py            # DP mechanisms
│   │
│   ├── preprocessing/
│   │   ├── feature_engineering.py
│   │   ├── normalization.py
│   │   └── imputation.py
│   │
│   └── utils/
│       ├── metrics.py
│       ├── visualization.py
│       └── config.py
│
├── mlops/
│   ├── mlflow_config.py
│   ├── experiment_tracking.py
│   └── model_registry.py
│
├── dashboard/
│   └── streamlit_app.py
│
├── terraform/
│   ├── main.tf
│   ├── vpc.tf
│   ├── ecs.tf
│   ├── s3.tf
│   └── variables.tf
│
├── docker/
│   ├── Dockerfile.server
│   ├── Dockerfile.client
│   ├── Dockerfile.mlflow
│   └── Dockerfile.dashboard
│
├── scripts/
│   ├── deploy.sh
│   ├── start_training.sh
│   └── cleanup.sh
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_results_analysis.ipynb
│
└── tests/
    ├── test_models.py
    ├── test_federated.py
    └── test_privacy.py
```

---

## 10. Timeline & Milestones

### Phase 1: Setup (Week 1)
- [ ] AWS account setup and IAM configuration
- [ ] Terraform infrastructure deployment
- [ ] Docker images built and pushed to ECR

### Phase 2: Data & Models (Week 2)
- [ ] Synthetic data generation
- [ ] Baseline model development
- [ ] Feature engineering pipeline

### Phase 3: Federated Implementation (Week 3)
- [ ] Flower server/client implementation
- [ ] Privacy mechanisms integration
- [ ] MLflow tracking setup

### Phase 4: Deployment & Testing (Week 4)
- [ ] ECS deployment
- [ ] End-to-end FL training
- [ ] Dashboard and monitoring

### Phase 5: Evaluation & Documentation (Week 5)
- [ ] Comprehensive experiments
- [ ] Results analysis
- [ ] Documentation and demo prep

---

## 11. Key Technologies

### Core ML Stack
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|----------|
| **FL Framework** | Flower | 1.7+ | Federated learning orchestration |
| **Deep Learning** | PyTorch | 2.2+ | Neural network training |
| **Distributed Training** | Ray | 2.9+ | Parallel execution, HPO |
| **Privacy** | Opacus | 1.4+ | Differential privacy for PyTorch |

### Data & Processing
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|----------|
| **Data Processing** | Pandas, Polars | Latest | Local data manipulation |
| **Big Data** | Apache Spark | 3.5+ | Large-scale preprocessing |
| **Data Lake** | Delta Lake | 3.0+ | Versioned data storage |
| **Catalog** | AWS Glue | - | Data catalog & ETL |
| **Query** | AWS Athena | - | SQL queries on S3 |

### MLOps & Monitoring
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|----------|
| **Experiment Tracking** | MLflow | 2.10+ | Metrics, models, experiments |
| **Model Registry** | MLflow + S3 | - | Model versioning |
| **Monitoring** | Prometheus + Grafana | Latest | Metrics & dashboards |
| **Logging** | CloudWatch / ELK | - | Centralized logs |
| **Alerting** | CloudWatch Alarms | - | Automated alerts |

### Cloud-Native ML
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|----------|
| **Data Prep** | SageMaker Processing | - | Distributed feature engineering |
| **Training** | SageMaker Training | - | Managed training jobs (baseline) |
| **Deployment** | SageMaker Endpoints | - | Scalable inference |
| **Batch Inference** | SageMaker Batch Transform | - | Large-scale predictions |
| **Feature Store** | SageMaker Feature Store | - | Online/offline features |
| **Orchestration** | Step Functions | - | Workflow automation |
| **Scheduling** | EventBridge | - | Cron-based triggers |

### Infrastructure
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|----------|
| **Orchestration** | Amazon EKS | 1.28+ | Kubernetes cluster |
| **Container Runtime** | Docker | 24+ | Containerization |
| **IaC** | Terraform | 1.7+ | Infrastructure as code |
| **Package Mgmt** | Helm | 3.13+ | Kubernetes packages |
| **Service Mesh** | Istio (optional) | 1.20+ | Advanced networking |
| **Secrets** | AWS Secrets Manager | - | Credential management |

### Visualization & UI
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|----------|
| **Dashboard** | Streamlit | 1.30+ | Real-time monitoring |
| **Notebooks** | JupyterHub | Latest | Interactive development |
| **Viz Libraries** | Plotly, Matplotlib | Latest | Data visualization |

### Development & CI/CD
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|----------|
| **Version Control** | Git | - | Code versioning |
| **CI/CD** | GitHub Actions | - | Automated testing & deployment |
| **Testing** | pytest | Latest | Unit & integration tests |
| **Code Quality** | Black, Ruff | Latest | Linting & formatting |
| **Container Registry** | Amazon ECR | - | Docker image storage |

---

## 12. Success Criteria

✅ **Functional**:
- FL system trains successfully across 3+ simulated hospitals
- Models converge within reasonable rounds (< 100)
- Automated deployment works end-to-end

✅ **Performance**:
- FL model achieves ≥85% of centralized baseline performance
- Privacy budget stays within acceptable limits (ε < 5)
- Training completes in < 2 hours

✅ **Scalability**:
- System handles 5+ hospitals without modification
- Graceful handling of client dropouts

✅ **Reproducibility**:
- Complete documentation
- One-command deployment
- Reproducible results (fixed seeds)

---

## 13. Future Extensions

1. **Real Healthcare Data**: Integration with MIMIC-IV or eICU
2. **Advanced Algorithms**: FedOpt, personalized FL, split learning
3. **Additional Modalities**: Medical notes (NLP), wearable data
4. **Production Features**: Model versioning, A/B testing, gradual rollout
5. **Multi-cloud**: Support for Azure, GCP
6. **Blockchain**: Immutable audit trail for model updates

---

## 14. References & Resources

### Frameworks & Tools
- **Flower Documentation**: https://flower.dev/docs/
- **Opacus (PyTorch DP)**: https://github.com/pytorch/opacus
- **MLflow**: https://mlflow.org/
- **Ray Documentation**: https://docs.ray.io/
- **Apache Spark**: https://spark.apache.org/docs/latest/
- **Amazon EKS Best Practices**: https://aws.github.io/aws-eks-best-practices/
- **Kubeflow**: https://www.kubeflow.org/docs/

### Datasets
- **Stroke Prediction Dataset**: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- **MIMIC-IV**: https://physionet.org/content/mimiciv/
- **eICU**: https://physionet.org/content/eicu-crd/
- **UCI ML Repository**: https://archive.ics.uci.edu/

### Research Papers

#### Federated Learning Foundations
- **McMahan et al. (2017)** - "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)
- **Li et al. (2020)** - "Federated Optimization in Heterogeneous Networks" (FedProx)
- **Reddi et al. (2021)** - "Adaptive Federated Optimization" (FedAdam, FedYogi)

#### Representation Learning
- **Chen et al. (2020)** - "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR)
- **He et al. (2020)** - "Momentum Contrast for Unsupervised Visual Representation Learning" (MoCo)
- **Vincent et al. (2008)** - "Extracting and Composing Robust Features with Denoising Autoencoders"

#### Data Fusion
- **Baltrusaitis et al. (2019)** - "Multimodal Machine Learning: A Survey and Taxonomy"
- **Ramachandram & Taylor (2017)** - "Deep Multimodal Learning: A Survey on Recent Advances"

#### Federated Learning + Healthcare
- **Rieke et al. (2020)** - "The Future of Digital Health with Federated Learning"
- **Xu et al. (2021)** - "Federated Learning for Healthcare Informatics"
- **Brisimi et al. (2018)** - "Federated Learning of Predictive Models from Federated Electronic Health Records"

#### Privacy
- **Dwork & Roth (2014)** - "The Algorithmic Foundations of Differential Privacy"
- **Abadi et al. (2016)** - "Deep Learning with Differential Privacy"
- **Bonawitz et al. (2017)** - "Practical Secure Aggregation for Privacy-Preserving Machine Learning"

---

**Document Version**: 1.0  
**Last Updated**: February 18, 2026  
**Author**: FL Demo Project Team
