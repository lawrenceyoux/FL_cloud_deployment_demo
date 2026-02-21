# Technical Specifications
## Federated Learning Healthcare Demo

> **Note**: This document describes the intended final-state technical specifications.
> The **current implementation** uses a much simpler stack:
> tabular Kaggle stroke data, a small MLP (`StrokeNet`), FedAvg, and MLflow.
> Multimodal data structures, differential privacy, and the full AWS/EKS stack described
> here are **not yet implemented**. See [README.md](README.md) for what currently works.

---

## 1. Data Model Specification

### 1.1 Patient Record Structure

```python
{
    "patient_id": "P_H1_0001",
    "hospital_id": "H1",
    "demographics": {
        "age": 65,
        "gender": "M",  # M/F
        "ethnicity": "Caucasian",
        "bmi": 28.5,
        "comorbidities": ["hypertension", "diabetes_type2"],
        "medications": ["metformin", "lisinopril"]
    },
    "clinical_timeseries": {
        "timestamps": ["2025-01-01", "2025-01-02", ..., "2025-12-31"],
        "heart_rate": [72, 75, 68, ...],      # beats/min
        "systolic_bp": [130, 128, 135, ...],   # mmHg
        "diastolic_bp": [85, 82, 88, ...],     # mmHg
        "spo2": [98, 97, 99, ...],             # %
        "temperature": [36.8, 36.9, 37.0, ...] # Celsius
    },
    "lab_results": {
        "timestamps": ["2025-01-07", "2025-01-14", ...],  # Weekly
        "wbc": [7.5, 7.2, 7.8, ...],           # 10^9/L
        "rbc": [4.8, 4.7, 4.9, ...],           # 10^12/L
        "hemoglobin": [14.5, 14.2, 14.7, ...], # g/dL
        "platelets": [250, 245, 260, ...],     # 10^9/L
        "glucose": [110, 105, 115, ...],       # mg/dL
        "creatinine": [1.0, 0.9, 1.1, ...],    # mg/dL
        "sodium": [140, 139, 141, ...],        # mEq/L
        "potassium": [4.2, 4.0, 4.3, ...]      # mEq/L
    },
    "biology_markers": {
        "timestamps": ["2025-01-15", "2025-02-15", ...],  # Monthly
        "crp": [2.5, 2.8, 2.3, ...],           # mg/L (inflammation)
        "d_dimer": [0.4, 0.5, 0.3, ...],       # mg/L (coagulation)
        "troponin": [0.01, 0.02, 0.01, ...],   # ng/mL (cardiac)
        "bnp": [80, 85, 75, ...],              # pg/mL (heart failure)
        "ldl": [120, 115, 125, ...]            # mg/dL (cholesterol)
    },
    "omics_data": {
        "gene_expression": {
            "APOE": 12.5,    # Alzheimer's risk
            "BRCA1": 8.2,    # Cancer susceptibility
            "IL6": 15.3,     # Inflammation
            "TNF": 10.1,     # Immune response
            # ... 50-100 genes total
        },
        "metabolomics": {
            "glucose_6_phosphate": 0.8,
            "lactate": 1.2,
            "pyruvate": 0.5,
            # ... 30-50 metabolites
        }
    },
    "outcomes": {
        "readmission_30d": 1,        # Binary: 0/1
        "risk_level": "High",        # Categorical: Low/Medium/High
        "days_to_readmission": 25,   # Continuous: days (or null)
        "adverse_event": 0           # Binary: 0/1
    },
    "metadata": {
        "admission_date": "2025-01-01",
        "discharge_date": "2025-01-10",
        "length_of_stay": 9,
        "diagnosis_codes": ["I50.9", "E11.9"],  # ICD-10
        "procedure_codes": ["0W9G30Z"]
    }
}
```

### 1.2 Data Distribution Across Hospitals

| Hospital | Patients | Avg Age | Male:Female | Readmission Rate | Data Characteristics |
|----------|----------|---------|-------------|------------------|----------------------|
| H1 | 1000 | 62 | 55:45 | 0.18 | Urban, diverse population |
| H2 | 800 | 58 | 48:52 | 0.15 | Suburban, younger cohort |
| H3 | 1200 | 68 | 60:40 | 0.22 | Rural, older, sicker patients |

**Non-IID Characteristics**:
- Different age distributions (systematic skew)
- Different readmission base rates
- Varying missingness patterns (rural hospital has less frequent labs)
- Different comorbidity prevalence

### 1.3 Feature Engineering Output

```python
# Processed feature vector dimensions
feature_dims = {
    "demographics": 10,           # One-hot + continuous
    "clinical_temporal": 256,     # LSTM embeddings
    "lab_temporal": 128,          # LSTM embeddings
    "biology_static": 20,         # Aggregated (mean, std, trend)
    "omics_reduced": 32,          # PCA-reduced
    "total": 446
}

# Target variables
targets = {
    "classification": 3,          # 3-class risk (Low/Med/High)
    "regression": 1,              # Days to readmission
    "clustering": None            # Unsupervised
}
```

---

## 2. Model Architecture Specifications

### 2.1 Multimodal Risk Classifier

```python
class MultimodalRiskClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Modality-specific encoders
        self.clinical_encoder = nn.LSTM(
            input_size=5,           # 5 vital signs
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        self.lab_encoder = nn.LSTM(
            input_size=8,           # 8 lab values
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        self.demographics_encoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )
        
        self.biology_encoder = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        self.omics_encoder = nn.Sequential(
            nn.Linear(32, 64),      # PCA-reduced input
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        # Attention-based fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)        # 3 risk classes
        )
    
    def forward(self, clinical, lab, demographics, biology, omics):
        # Encode each modality
        clinical_emb, _ = self.clinical_encoder(clinical)
        clinical_emb = clinical_emb[:, -1, :]  # Last timestep
        
        lab_emb, _ = self.lab_encoder(lab)
        lab_emb = lab_emb[:, -1, :]
        
        demo_emb = self.demographics_encoder(demographics)
        bio_emb = self.biology_encoder(biology)
        omics_emb = self.omics_encoder(omics)
        
        # Concatenate embeddings
        combined = torch.cat([
            clinical_emb,
            lab_emb,
            demo_emb,
            bio_emb,
            omics_emb
        ], dim=1)
        
        # Attention fusion
        combined = combined.unsqueeze(0)
        fused, _ = self.attention(combined, combined, combined)
        fused = fused.squeeze(0)
        
        # Classification
        logits = self.classifier(fused)
        return logits

# Model size
total_params = 587_392
trainable_params = 587_392
model_size_mb = 2.24
```

### 2.2 Federated Clustering Model

```python
class FederatedAutoencoder(nn.Module):
    """Autoencoder for representation learning in FL setting"""
    def __init__(self, input_dim=446, latent_dim=32):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim)
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# After FL training, perform clustering on latent representations
# Use K-Means or Gaussian Mixture Models
```

---

## 3. Federated Learning Configuration

### 3.1 Server Configuration

```yaml
server:
  address: "0.0.0.0:8080"
  num_rounds: 50
  min_available_clients: 3
  
  strategy:
    name: "FedAvg"
    fraction_fit: 1.0           # Use all clients for training
    fraction_evaluate: 1.0      # Use all clients for evaluation
    min_fit_clients: 3
    min_evaluate_clients: 3
    
  aggregation:
    method: "weighted_average"  # Weight by dataset size
    clip_norm: 1.0              # Gradient clipping for DP
    
  model:
    type: "MultimodalRiskClassifier"
    checkpoint_freq: 5          # Save every 5 rounds
    early_stopping_patience: 10
```

### 3.2 Client Configuration

```yaml
client:
  hospital_id: "H1"             # Unique identifier
  data_path: "s3://fl-demo-data-hospital-1/"
  
  training:
    local_epochs: 5
    batch_size: 32
    learning_rate: 0.001
    optimizer: "Adam"
    loss_function: "CrossEntropyLoss"
    
  privacy:
    enable_dp: true
    noise_multiplier: 1.1       # Controls privacy-utility tradeoff
    max_grad_norm: 1.0
    target_epsilon: 1.0
    target_delta: 1e-5
    
  hardware:
    device: "cuda"              # Use GPU if available
    num_workers: 4              # DataLoader workers
    pin_memory: true
```

### 3.3 Hyperparameter Search Space

```python
hyperparameters = {
    "learning_rate": [0.0001, 0.0005, 0.001, 0.005],
    "batch_size": [16, 32, 64],
    "local_epochs": [3, 5, 10],
    "hidden_dim": [64, 128, 256],
    "dropout": [0.1, 0.2, 0.3],
    "privacy_epsilon": [0.5, 1.0, 2.0, 5.0, float('inf')],
    "aggregation_strategy": ["FedAvg", "FedProx", "FedAdam"],
}

# Use Optuna or Ray Tune for hyperparameter optimization
```

---

## 4. Privacy Specifications

### 4.1 Differential Privacy Parameters

```python
privacy_config = {
    "mechanism": "Gaussian",         # Noise distribution
    "epsilon": 1.0,                  # Privacy budget
    "delta": 1e-5,                   # Failure probability
    "max_grad_norm": 1.0,            # L2 clipping threshold
    "noise_multiplier": 1.1,         # σ = noise_multiplier * sensitivity
    "secure_mode": True,             # Use cryptographic PRNG
    "accountant": "RDP",             # Rényi Differential Privacy
}

# Privacy amplification by subsampling
subsampling_rate = 0.01              # 1% of data per batch
amplified_epsilon = epsilon / sqrt(subsampling_rate)

# Privacy budget allocation
epsilon_per_round = total_epsilon / num_rounds
```

### 4.2 Secure Aggregation

```python
secure_aggregation = {
    "enabled": True,
    "protocol": "SecAgg",            # Bonawitz et al. (2017)
    "min_clients_for_decrypt": 3,    # Threshold
    "encryption": "Paillier",        # Homomorphic encryption
    "key_size": 2048,                # bits
}

# Communication protocol
communication = {
    "encryption": "TLS 1.3",
    "authentication": "mTLS",         # Mutual TLS
    "certificate_authority": "AWS ACM",
}
```

---

## 5. Infrastructure Specifications

### 5.1 AWS Resources

#### Compute (ECS Fargate)

```yaml
FL_Server:
  task_cpu: 2048                # 2 vCPU
  task_memory: 4096             # 4 GB
  desired_count: 1
  auto_scaling:
    min: 1
    max: 3
    target_cpu: 70%

FL_Client:
  task_cpu: 1024                # 1 vCPU
  task_memory: 2048             # 2 GB
  desired_count: 3              # 3 hospitals
  auto_scaling:
    min: 3
    max: 10
    target_cpu: 70%

MLflow_Server:
  task_cpu: 1024
  task_memory: 2048
  desired_count: 1

Dashboard:
  task_cpu: 512
  task_memory: 1024
  desired_count: 1
```

#### Storage (S3)

```yaml
S3_Buckets:
  fl-demo-data-hospital-1:
    size: 500 MB
    versioning: enabled
    encryption: AES-256
    lifecycle:
      transition_to_ia: 30 days
      expire: never
  
  fl-demo-data-hospital-2:
    size: 400 MB
    # ... same config
  
  fl-demo-data-hospital-3:
    size: 600 MB
    # ... same config
  
  fl-demo-models:
    size: 2 GB
    versioning: enabled
    encryption: AES-256
    lifecycle:
      transition_to_ia: 90 days
      expire: 365 days
  
  fl-demo-artifacts:
    size: 5 GB
    encryption: AES-256
```

#### Database (DynamoDB)

```yaml
DynamoDB_Tables:
  fl-training-metadata:
    partition_key: run_id
    sort_key: round_number
    attributes:
      - run_id (String)
      - round_number (Number)
      - timestamp (String)
      - global_accuracy (Number)
      - privacy_budget (Number)
      - participating_clients (List)
    billing_mode: PAY_PER_REQUEST
    
  fl-client-status:
    partition_key: client_id
    sort_key: timestamp
    ttl: 7 days
```

#### Networking

```yaml
VPC:
  cidr: 10.0.0.0/16
  
  public_subnets:
    - 10.0.1.0/24  (us-east-1a)
    - 10.0.2.0/24  (us-east-1b)
  
  private_subnets:
    - 10.0.10.0/24 (us-east-1a)
    - 10.0.11.0/24 (us-east-1b)
  
  nat_gateway: true
  
  security_groups:
    fl_server_sg:
      ingress:
        - port: 8080
          source: client_sg
      egress:
        - port: 443 (HTTPS)
        - port: 5000 (MLflow)
    
    fl_client_sg:
      ingress:
        - port: 22 (SSH, from bastion only)
      egress:
        - port: 8080 (to server)
        - port: 443 (S3, CloudWatch)
```

### 5.2 Cost Breakdown (Monthly)

```
ECS Fargate:
  FL Server (2 vCPU, 4 GB, 24/7):         $35
  FL Clients (3x 1 vCPU, 2 GB, 8h/day):   $30
  MLflow Server (1 vCPU, 2 GB, 24/7):     $18
  Dashboard (0.5 vCPU, 1 GB, 24/7):       $9
  Subtotal:                                $92

Storage:
  S3 Storage (8 GB):                       $0.18
  S3 Requests (1M PUT/GET):                $5
  DynamoDB (on-demand):                    $5
  Subtotal:                                $10.18

Networking:
  NAT Gateway (1 gateway, 24/7):           $32
  Data Transfer (100 GB out):              $9
  ALB (Application Load Balancer):         $16
  Subtotal:                                $57

Monitoring:
  CloudWatch Logs (10 GB):                 $5
  CloudWatch Metrics (custom):             $3
  Subtotal:                                $8

-------------------------------------------
TOTAL MONTHLY COST:                        $167.18

Cost Optimizations:
  - Use Spot instances for clients:        -$21 (save 70% on clients)
  - Shutdown off-hours (16h/day):          -$45 (save 50% on compute)
  - Use S3 Intelligent Tiering:            -$2
  - Optimized NAT (VPC endpoints):         -$25
  
OPTIMIZED MONTHLY COST:                    ~$74
```

---

## 6. Monitoring & Metrics

### 6.1 MLflow Metrics

```python
mlflow_metrics = {
    # Model performance
    "train_loss": "float",
    "train_accuracy": "float",
    "val_loss": "float",
    "val_accuracy": "float",
    "test_accuracy": "float",
    "auc_roc": "float",
    "auc_pr": "float",
    "f1_score": "float",
    "precision": "float",
    "recall": "float",
    
    # Per-class metrics
    "accuracy_high_risk": "float",
    "accuracy_medium_risk": "float",
    "accuracy_low_risk": "float",
    
    # Federated learning metrics
    "num_participating_clients": "int",
    "aggregation_time_seconds": "float",
    "communication_bytes": "int",
    "round_duration_seconds": "float",
    
    # Privacy metrics
    "privacy_epsilon": "float",
    "privacy_delta": "float",
    "noise_multiplier": "float",
    
    # Fairness metrics
    "performance_variance": "float",   # Across hospitals
    "min_hospital_accuracy": "float",
    "max_hospital_accuracy": "float",
}

# Logged every round
mlflow.log_metrics(mlflow_metrics, step=round_num)
```

### 6.2 CloudWatch Custom Metrics

```python
cloudwatch_metrics = {
    "Namespace": "FederatedLearning",
    "Metrics": [
        {
            "MetricName": "TrainingRound",
            "Value": round_number,
            "Unit": "Count"
        },
        {
            "MetricName": "GlobalModelAccuracy",
            "Value": accuracy,
            "Unit": "Percent"
        },
        {
            "MetricName": "ActiveClients",
            "Value": num_active_clients,
            "Unit": "Count"
        },
        {
            "MetricName": "PrivacyBudget",
            "Value": epsilon_consumed,
            "Unit": "None"
        },
        {
            "MetricName": "ModelSize",
            "Value": model_size_mb,
            "Unit": "Megabytes"
        },
        {
            "MetricName": "CommunicationOverhead",
            "Value": bytes_transferred,
            "Unit": "Bytes"
        }
    ]
}
```

### 6.3 Alarms & Alerts

```yaml
CloudWatch_Alarms:
  training_failures:
    metric: "TrainingRound"
    threshold: 0
    evaluation_periods: 2
    datapoints_to_alarm: 2
    action: SNS notification
  
  low_accuracy:
    metric: "GlobalModelAccuracy"
    threshold: 0.60
    comparison: "LessThanThreshold"
    action: SNS notification
  
  privacy_budget_exceeded:
    metric: "PrivacyBudget"
    threshold: 5.0
    comparison: "GreaterThanThreshold"
    action: Stop training, SNS notification
  
  client_dropout:
    metric: "ActiveClients"
    threshold: 2
    comparison: "LessThanThreshold"
    action: SNS notification, Auto-scale
```

---

## 7. API Specifications

### 7.1 FL Server API

```python
# gRPC service definition
service FederatedLearning {
    rpc GetParameters (ClientInfo) returns (ModelParameters);
    rpc UpdateParameters (ModelUpdate) returns (Acknowledgment);
    rpc EvaluateModel (ModelParameters) returns (EvaluationResult);
    rpc GetStatus (Empty) returns (ServerStatus);
}

message ClientInfo {
    string client_id = 1;
    int32 num_samples = 2;
    string version = 3;
}

message ModelParameters {
    repeated float weights = 1;
    int32 round_number = 2;
    map<string, float> config = 3;
}

message ModelUpdate {
    string client_id = 1;
    repeated float gradients = 2;
    int32 num_samples = 3;
    map<string, float> metrics = 4;
}
```

### 7.2 REST API (for dashboard/monitoring)

```yaml
REST_Endpoints:
  GET /api/v1/status:
    description: Get current training status
    response:
      - current_round: int
      - total_rounds: int
      - active_clients: list[str]
      - global_accuracy: float
      - privacy_budget: float
  
  GET /api/v1/metrics:
    description: Get training metrics
    parameters:
      - run_id: str
      - round_number: int (optional)
    response:
      - metrics: dict
      - timestamp: str
  
  POST /api/v1/train/start:
    description: Start new training run
    body:
      - config: dict
    response:
      - run_id: str
      - status: str
  
  POST /api/v1/train/stop:
    description: Stop current training run
    parameters:
      - run_id: str
    response:
      - status: str
```

---

## 8. Testing Specifications

### 8.1 Unit Tests

```python
# tests/test_models.py
def test_multimodal_classifier_forward_pass():
    # Test model can process batch
    
def test_model_serialization():
    # Test model save/load
    
def test_gradient_computation():
    # Test backpropagation works

# tests/test_federated.py
def test_fedavg_aggregation():
    # Test weighted averaging
    
def test_client_server_communication():
    # Test gRPC communication
    
def test_differential_privacy():
    # Test DP mechanism

# tests/test_privacy.py
def test_privacy_budget_tracking():
    # Test epsilon accounting
    
def test_gradient_clipping():
    # Test L2 norm clipping
```

### 8.2 Integration Tests

```python
# tests/test_integration.py
def test_end_to_end_federated_training():
    # Spin up server + 3 clients
    # Run 5 rounds
    # Verify convergence
    
def test_s3_data_access():
    # Test clients can read from S3
    
def test_mlflow_logging():
    # Test metrics are logged correctly
```

### 8.3 Performance Tests

```python
# tests/test_performance.py
def test_training_speed():
    # Measure time per round
    # Should be < 2 minutes per round
    
def test_communication_overhead():
    # Measure bytes transferred
    # Should be < 10 MB per round per client
    
def test_scalability():
    # Test with 5, 10, 15 clients
    # Verify linear scaling
```

---

## 9. Deployment Checklist

### Pre-Deployment
- [ ] AWS credentials configured
- [ ] Terraform state backend configured (S3 + DynamoDB)
- [ ] Domain name registered (optional)
- [ ] SSL certificates provisioned
- [ ] Budget alerts configured
- [ ] IAM roles reviewed for least privilege

### Deployment
- [ ] Run `terraform apply`
- [ ] Build and push Docker images to ECR
- [ ] Deploy ECS services
- [ ] Upload synthetic data to S3
- [ ] Test connectivity (server ↔ clients)
- [ ] Verify MLflow tracking server accessible
- [ ] Configure CloudWatch alarms

### Post-Deployment
- [ ] Run smoke tests
- [ ] Monitor initial training run
- [ ] Verify metrics logged to MLflow
- [ ] Check CloudWatch logs for errors
- [ ] Test dashboard accessibility
- [ ] Document any configuration changes

### Teardown
- [ ] Stop all ECS tasks
- [ ] Delete S3 buckets (if desired)
- [ ] Run `terraform destroy`
- [ ] Verify no resources left (cost check)

---

## 10. Security Considerations

### Data Security
- ✅ Encryption at rest (S3 AES-256)
- ✅ Encryption in transit (TLS 1.3)
- ✅ VPC isolation (private subnets)
- ✅ IAM least privilege
- ✅ S3 bucket policies (deny public access)
- ✅ Secrets in AWS Secrets Manager

### Model Security
- ✅ Differential privacy (prevents data leakage)
- ✅ Secure aggregation (encrypted model updates)
- ✅ Model versioning (audit trail)
- ✅ Checksum validation (detect tampering)

### Network Security
- ✅ Security groups (port restrictions)
- ✅ NACLs (network-level firewall)
- ✅ AWS WAF (web application firewall for ALB)
- ✅ VPC Flow Logs (network monitoring)

### Compliance
- ✅ HIPAA-aligned architecture
- ✅ Audit logging (CloudTrail, CloudWatch)
- ✅ Data retention policies
- ✅ Disaster recovery plan
- ✅ Incident response plan

---

**Document Version**: 1.0  
**Last Updated**: February 18, 2026
