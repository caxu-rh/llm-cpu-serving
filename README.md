# Serve a lightweight HR assistant

![chat-example.png](docs/images/chat-example.png)

Replace hours spent searching policy documents with higher-value relational work.

## Detailed description

The *Assistant to the HR Representative* is a lightweight quickstart designed to
give HR Representatives in Financial Services a trusted sounding board for discussions and decisions.
Chat with this assistant for quick insights and actionable advice.

This quickstart was designed for environments where GPUs are not available or
necessary, making it ideal for lightweight inference use cases, prototyping, or
constrained environments. By making the most of vLLM on CPU-based
infrastructure, this Assistant to the HR Representative can be deployed to almost any OpenShift AI
environment.

This quickstart includes a Helm chart for deploying:

- An OpenShift AI Project.
- vLLM with CPU support running Facebook's OPT-125m model (125M parameters).
- AnythingLLM, a versatile chat interface, running as a workbench and connected
  to the vLLM inference service.

Use this project to quickly spin up a minimal vLLM instance and start serving
lightweight models like OPT-125m on CPU—no GPU required. 🚀

**Model Info:** This deployment uses [facebook/opt-125m](https://huggingface.co/facebook/opt-125m),
a 125M parameter model optimized for fast CPU inference. It's approximately 8-10x faster than
larger models like TinyLlama while maintaining good quality for chat and Q&A tasks.


<!-- ### See it in action

Red Hat uses Arcade software to create interactive demos. Check out 
[Quickstart with TinyLlama on CPU](https://interact.redhat.com/share/zsT3j9cgPt9yyPchb7EJ)
 to see it in action. -->


#### Detailed Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          OpenShift AI / OpenShift Cluster                       │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    Namespace: hr-assistant                              │   │
│  │                                                                         │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │   │
│  │  │  User Interface Layer                                            │  │   │
│  │  │  ┌────────────────────────────────────────────────────────────┐  │  │   │
│  │  │  │  Data Science Gateway (OpenShift Route)                    │  │  │   │
│  │  │  │  https://data-science-gateway.apps.../hr-assistant/...     │  │  │   │
│  │  │  └─────────────────────┬──────────────────────────────────────┘  │  │   │
│  │  │                        │                                          │  │   │
│  │  │                        ▼                                          │  │   │
│  │  │  ┌────────────────────────────────────────────────────────────┐  │  │   │
│  │  │  │  AnythingLLM Workbench (StatefulSet)                       │  │  │   │
│  │  │  │  Pod: anythingllm-0                                        │  │  │   │
│  │  │  │  ┌──────────────────────────────────────────────────────┐  │  │  │   │
│  │  │  │  │  Container: kube-rbac-proxy (auto-injected)          │  │  │  │   │
│  │  │  │  │  Port: 8443 (HTTPS with RBAC authentication)         │  │  │  │   │
│  │  │  │  │  Note: Injected by OpenShift AI controller           │  │  │  │   │
│  │  │  │  └───────────────────┬──────────────────────────────────┘  │  │  │   │
│  │  │  │                      │                                      │  │  │   │
│  │  │  │  ┌───────────────────▼──────────────────────────────────┐  │  │  │   │
│  │  │  │  │  Container: anythingllm                              │  │  │  │   │
│  │  │  │  │  Port: 8888 (Jupyter/AnythingLLM interface)          │  │  │  │   │
│  │  │  │  │                                                       │  │  │  │   │
│  │  │  │  │  Features:                                           │  │  │  │   │
│  │  │  │  │  • Chat interface for end users                      │  │  │  │   │
│  │  │  │  │  • Document embedding (native embedder)              │  │  │  │   │
│  │  │  │  │  • Vector database (LanceDB)                         │  │  │  │   │
│  │  │  │  │  • RAG (Retrieval-Augmented Generation)              │  │  │  │   │
│  │  │  │  │  • Workspace: "Assistant to the HR Representative"   │  │  │  │   │
│  │  │  │  │                                                       │  │  │  │   │
│  │  │  │  │  Environment (from Secret: opt-125m-vllm-cpu):       │  │  │  │   │
│  │  │  │  │  • LLM_PROVIDER: generic-openai                      │  │  │  │   │
│  │  │  │  │  • GENERIC_OPEN_AI_BASE_PATH:                        │  │  │  │   │
│  │  │  │  │      http://opt-125m-cpu-predictor:8080/v1           │  │  │  │   │
│  │  │  │  │  • GENERIC_OPEN_AI_MODEL_PREF: opt-125m              │  │  │  │   │
│  │  │  │  │  • EMBEDDING_ENGINE: native                          │  │  │  │   │
│  │  │  │  │  • VECTOR_DB: lancedb                                │  │  │  │   │
│  │  │  │  └───────────────────┬──────────────────────────────────┘  │  │  │   │
│  │  │  │                      │                                      │  │  │   │
│  │  │  │  ┌───────────────────▼──────────────────────────────────┐  │  │  │   │
│  │  │  │  │  Container: anythingllm-automation (sidecar)         │  │  │  │   │
│  │  │  │  │  • SQLite database management                        │  │  │  │   │
│  │  │  │  │  • API key setup automation                          │  │  │  │   │
│  │  │  │  └──────────────────────────────────────────────────────┘  │  │  │   │
│  │  │  │                                                             │  │  │   │
│  │  │  │  Volumes:                                                   │  │  │   │
│  │  │  │  • PVC: anythingllm (persistent storage)                    │  │  │   │
│  │  │  │  • ConfigMap: workbench-trusted-ca-bundle                   │  │  │   │
│  │  │  │  • Secret: anythingllm-kube-rbac-proxy-tls (auto-created)   │  │  │   │
│  │  │  │  • ConfigMap: anythingllm-kube-rbac-proxy-config (auto)     │  │  │   │
│  │  │  └─────────────────────────────────────────────────────────────┘  │  │   │
│  │  │                        │                                          │  │   │
│  │  │                        │ HTTP POST /v1/chat/completions           │  │   │
│  │  │                        │ (OpenAI-compatible API calls)            │  │   │
│  │  │                        │                                          │  │   │
│  │  └────────────────────────┼──────────────────────────────────────────┘  │   │
│  │                           │                                             │   │
│  │                           ▼                                             │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Inference Service Layer                                         │  │   │
│  │  │  ┌────────────────────────────────────────────────────────────┐  │  │   │
│  │  │  │  Service: opt-125m-cpu-predictor                           │  │  │   │
│  │  │  │  Type: Headless (ClusterIP: None)                          │  │  │   │
│  │  │  │  Port: 80 → Target: 8080                                   │  │  │   │
│  │  │  └─────────────────────┬──────────────────────────────────────┘  │  │   │
│  │  │                        │                                          │  │   │
│  │  │                        ▼                                          │  │   │
│  │  │  ┌────────────────────────────────────────────────────────────┐  │  │   │
│  │  │  │  InferenceService: opt-125m-cpu (KServe)                   │  │  │   │
│  │  │  │  Deployment Mode: RawDeployment                            │  │  │   │
│  │  │  │  Runtime: vllm-cpu (ServingRuntime)                        │  │  │   │
│  │  │  │                                                            │  │  │   │
│  │  │  │  Pod: opt-125m-cpu-predictor-xxxxxxxxx-xxxxx               │  │  │   │
│  │  │  │  ┌──────────────────────────────────────────────────────┐  │  │  │   │
│  │  │  │  │  Container: agent (KServe Agent)                     │  │  │  │   │
│  │  │  │  │  • Model loading and lifecycle management            │  │  │  │   │
│  │  │  │  │  • Health checks and monitoring                      │  │  │  │   │
│  │  │  │  └───────────────────┬──────────────────────────────────┘  │  │  │   │
│  │  │  │                      │                                      │  │  │   │
│  │  │  │  ┌───────────────────▼──────────────────────────────────┐  │  │  │   │
│  │  │  │  │  Container: kserve-container (vLLM)                  │  │  │  │   │
│  │  │  │  │  Port: 8080 (HTTP)                                   │  │  │  │   │
│  │  │  │  │                                                       │  │  │  │   │
│  │  │  │  │  vLLM Server Configuration:                          │  │  │  │   │
│  │  │  │  │  • Model: facebook/opt-125m (from HuggingFace)       │  │  │  │   │
│  │  │  │  │  • Dtype: float32 (CPU optimized)                    │  │  │  │   │
│  │  │  │  │  • Max model length: 2048 tokens                     │  │  │  │   │
│  │  │  │  │  • Served model name: opt-125m                       │  │  │  │   │
│  │  │  │  │  • Chat template: /app/chat-template/template.jinja  │  │  │  │   │
│  │  │  │  │                                                       │  │  │  │   │
│  │  │  │  │  Environment Variables:                              │  │  │  │   │
│  │  │  │  │  • VLLM_CPU_DISABLE_AVX512=1                         │  │  │  │   │
│  │  │  │  │  • ONEDNN_VERBOSE=0                                  │  │  │  │   │
│  │  │  │  │                                                       │  │  │  │   │
│  │  │  │  │  API Endpoints:                                      │  │  │  │   │
│  │  │  │  │  • GET  /health                                      │  │  │  │   │
│  │  │  │  │  • GET  /v1/models                                   │  │  │  │   │
│  │  │  │  │  • POST /v1/chat/completions ← (Primary)            │  │  │  │   │
│  │  │  │  │  • POST /v1/completions                             │  │  │  │   │
│  │  │  │  │  • POST /v1/embeddings                              │  │  │  │   │
│  │  │  │  │                                                       │  │  │  │   │
│  │  │  │  │  Resources:                                          │  │  │  │   │
│  │  │  │  │  • Requests: 2 CPU, 4Gi memory                       │  │  │  │   │
│  │  │  │  │  • Limits: 8 CPU, 8Gi memory                         │  │  │  │   │
│  │  │  │  └───────────────────┬──────────────────────────────────┘  │  │  │   │
│  │  │  │                      │                                      │  │  │   │
│  │  │  │                      ▼                                      │  │  │   │
│  │  │  │  ┌──────────────────────────────────────────────────────┐  │  │  │   │
│  │  │  │  │  Volume: chat-template (ConfigMap)                   │  │  │  │   │
│  │  │  │  │  Mounted at: /app/chat-template/                     │  │  │  │   │
│  │  │  │  │                                                       │  │  │  │   │
│  │  │  │  │  template.jinja:                                     │  │  │  │   │
│  │  │  │  │  Custom Jinja2 template for chat formatting          │  │  │  │   │
│  │  │  │  │  (Converts messages to prompt for OPT-125m)          │  │  │  │   │
│  │  │  │  └──────────────────────────────────────────────────────┘  │  │  │   │
│  │  │  └────────────────────────────────────────────────────────────┘  │  │   │
│  │  └──────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                         │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Supporting Resources                                            │  │   │
│  │  │                                                                  │  │   │
│  │  │  Helm-Managed Resources:                                         │  │   │
│  │  │  ├── ConfigMaps:                                                 │  │   │
│  │  │  │   • vllm-chat-template - Chat template for model             │  │   │
│  │  │  │   • workbench-trusted-ca-bundle - CA certificates            │  │   │
│  │  │  │   • modelconfig-opt-125m-cpu-0 - Model configuration         │  │   │
│  │  │  ├── Secrets:                                                    │  │   │
│  │  │  │   • opt-125m-vllm-cpu - AnythingLLM LLM provider config      │  │   │
│  │  │  │   • anythingllm-api - API key for AnythingLLM                │  │   │
│  │  │  ├── ServiceAccounts:                                            │  │   │
│  │  │  │   • anythingllm - Identity for AnythingLLM pod               │  │   │
│  │  │  ├── ServingRuntime:                                             │  │   │
│  │  │  │   • vllm-cpu - Defines vLLM container spec                   │  │   │
│  │  │  └── Jobs:                                                       │  │   │
│  │  │      • anythingllm-seed - Pre-seeds workspace with documents    │  │   │
│  │  │                                                                  │  │   │
│  │  │  Auto-Created by OpenShift AI Controller:                        │  │   │
│  │  │  ├── Services (with ownerReferences):                            │  │   │
│  │  │  │   • anythingllm - Port 80→8888 (main workbench)              │  │   │
│  │  │  │   • anythingllm-kube-rbac-proxy - Port 8443 (auth proxy)     │  │   │
│  │  │  ├── HTTPRoute (in redhat-ods-applications namespace):           │  │   │
│  │  │  │   • nb-hr-assistant-anythingllm                              │  │   │
│  │  │  │     Backend: anythingllm-kube-rbac-proxy:8443                │  │   │
│  │  │  ├── ReferenceGrant:                                             │  │   │
│  │  │  │   • notebook-httproute-access (cross-namespace access)       │  │   │
│  │  │  ├── ConfigMaps:                                                 │  │   │
│  │  │  │   • anythingllm-kube-rbac-proxy-config                       │  │   │
│  │  │  └── Secrets:                                                    │  │   │
│  │  │      • anythingllm-kube-rbac-proxy-tls (TLS certificates)       │  │   │
│  │  └──────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  External Dependencies:                                                         │
│  • HuggingFace Hub: facebook/opt-125m model download                           │
│  • Red Hat OpenShift Service Mesh: Networking and routing                      │
│  • Red Hat OpenShift Serverless (KServe): Model serving platform               │
└─────────────────────────────────────────────────────────────────────────────────┘

Request Flow:
═════════════

1. User Request Flow:
   User → Data Science Gateway → HTTPRoute → kube-rbac-proxy → AnythingLLM UI

2. Chat Message Processing:
   a. User sends message in AnythingLLM
   b. AnythingLLM embeds the query (native embedder)
   c. AnythingLLM searches vector DB (LanceDB) for relevant context
   d. AnythingLLM constructs chat completion request with context

3. Inference Request:
   AnythingLLM → POST http://opt-125m-cpu-predictor:8080/v1/chat/completions

   Request Body:
   {
     "model": "opt-125m",
     "messages": [
       {"role": "system", "content": "System prompt with HR context..."},
       {"role": "user", "content": "User question..."}
     ],
     "max_tokens": 512,
     "temperature": 0.7
   }

4. vLLM Processing:
   a. Receives request at /v1/chat/completions endpoint
   b. Applies chat template (template.jinja) to convert messages to prompt
   c. Tokenizes prompt using OPT-125m tokenizer
   d. Runs inference on CPU (float32 dtype)
   e. Generates tokens autoregressively
   f. Returns streaming or complete response

5. Response Path:
   vLLM → opt-125m-cpu-predictor Service → AnythingLLM → User Interface

Performance Characteristics:
════════════════════════════

• Model Size: 125M parameters (~500MB on disk)
• Inference Speed: ~20-25 seconds for 50 tokens on 8 CPU cores
• Memory Usage: 2-4GB active, 4-8GB total with cache
• Throughput: ~2-3 tokens/second on CPU
• Max Context: 2048 tokens
• Concurrency: Supports multiple requests (CPU KV cache managed by vLLM)
```

## Requirements 


### Minimum hardware requirements

- No GPU needed! 🤖
- 2 CPU cores
- 4 Gi memory
- Storage: 5Gi

**Note:** OPT-125m is lightweight and can run on minimal hardware. Response times will be slower with minimum resources.

### Recommended hardware requirements

- No GPU needed! 🤖
- 8 CPU cores
- 8 Gi memory
- Storage: 5Gi

**Note:** This configuration provides optimal performance with ~2-3 tokens/second generation speed.

### CPU Architecture Notes

This version is compiled for Intel CPUs (preferably with AVX512 enabled for better performance, but optional).
We disable AVX512 BRGEMM optimizations by default for stability (`VLLM_CPU_DISABLE_AVX512=1`).

Example AWS machine that works well: [m6i.4xlarge](https://instances.vantage.sh/aws/ec2/m6i.4xlarge) (16 vCPU, 64 GiB)

### Minimum software requirements

- Red Hat OpenShift 4.16.24 or later
- Red Hat OpenShift AI 2.16.2 or later
- Dependencies for [Single-model server](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.16/html/installing_and_uninstalling_openshift_ai_self-managed/installing-the-single-model-serving-platform_component-install#configuring-automated-installation-of-kserve_component-install):
    - Red Hat OpenShift Service Mesh
    - Red Hat OpenShift Serverless

### Required user permissions

- Standard user. No elevated cluster permissions required.


## Deploy

Follow the below steps to deploy and test the HR assistant.

### Prerequisites

Before deploying, ensure the following are available on your OpenShift cluster:

#### 1. OpenShift AI Installed and Configured
- Red Hat OpenShift AI 2.16.2 or later must be installed
- Single-model serving platform components must be configured:
  - Red Hat OpenShift Service Mesh
  - Red Hat OpenShift Serverless (KServe)

#### 2. Data Science Gateway
- Verify the gateway exists:
  ```bash
  oc get gateway data-science-gateway -n openshift-ingress
  ```
  Expected output should show the gateway in `PROGRAMMED` state.

#### 3. AnythingLLM ImageStream (REQUIRED)
The workbench references an ImageStream that must exist in the `redhat-ods-applications` namespace:

```bash
# Check if ImageStream exists
oc get imagestream custom-anythingllm -n redhat-ods-applications
```

**If the ImageStream does NOT exist**, create it before deploying:

```bash
cat <<EOF | oc apply -f -
apiVersion: image.openshift.io/v1
kind: ImageStream
metadata:
  name: custom-anythingllm
  namespace: redhat-ods-applications
  labels:
    opendatahub.io/notebook-image: "true"
spec:
  lookupPolicy:
    local: true
  tags:
    - name: "1.9.1"
      from:
        kind: DockerImage
        name: quay.io/rh-aiservices-bu/anythingllm-workbench:1.9.1
      importPolicy:
        scheduled: true
      referencePolicy:
        type: Local
EOF
```

**Note:** This ImageStream is required because the Notebook workbench references it, and OpenShift AI expects workbench images to be available as ImageStreams.

#### 4. Storage Class
Verify your cluster has a compatible storage class. The default in `helm/values.yaml` is:

```yaml
storageClassName: ocs-external-storagecluster-ceph-rbd
```

To use a different storage class, update `helm/values.yaml` before deploying:

```bash
# List available storage classes
oc get storageclass

# Update values.yaml if needed
```

Common storage class names:
- OpenShift Container Storage: `ocs-external-storagecluster-ceph-rbd`
- AWS EBS: `gp3-csi`, `gp2`
- Azure Disk: `managed-premium`
- GCP PD: `standard-rwo`

### Portability Checklist

When deploying to a new cluster, verify:

- ✅ OpenShift AI is installed and Data Science Gateway is running
- ✅ `custom-anythingllm` ImageStream exists in `redhat-ods-applications` namespace
- ✅ Storage class in `helm/values.yaml` matches your cluster's available storage
- ✅ Cluster has sufficient resources (8 CPU cores, 8Gi memory recommended)

**Quick Verification:** Run the prerequisites check script:

```bash
./scripts/verify-prerequisites.sh
```

This script will verify all requirements and provide specific instructions if anything is missing.

Follow the below steps to deploy and test the HR assistant.

### Clone

```bash
git clone https://github.com/rocrisp/llm-cpu-serving.git && \
    cd llm-cpu-serving/
```

**Note:** This is a fork of the original [rh-ai-quickstart/llm-cpu-serving](https://github.com/rh-ai-quickstart/llm-cpu-serving) with optimizations for OPT-125m on CPU. See [CHANGES.md](CHANGES.md) for detailed modifications.

<!-- ### (Optional) Update storage class name

If needed, update storage class name in `helm/values.yaml`.
```
storageClassName: gp3-csi
``` -->

### Create the project

```bash
PROJECT="hr-assistant"

oc new-project ${PROJECT}
``` 

### Install with Helm

```
helm install ${PROJECT} helm/ --namespace  ${PROJECT} 
```

### Wait for pods

```bash
oc -n ${PROJECT} get pods -w
```

Wait until all pods are in `Running` or `Completed` status:

```
(Expected output)
NAME                                      READY   STATUS      RESTARTS   AGE
anythingllm-0                             3/3     Running     0          2m
anythingllm-seed-xxxxx                    0/1     Completed   0          2m
opt-125m-cpu-predictor-xxxxxxxxxx-xxxxx   2/2     Running     0          2m
```

**Note:** The vLLM predictor pod may take 30-60 seconds to become ready as it downloads the OPT-125m model from HuggingFace (~500MB) on first start.

### Test

You can get the OpenShift AI Dashboard URL by:
```bash
# Get the Data Science Gateway route (main access point for OpenShift AI)
oc get route data-science-gateway -n openshift-ingress -o jsonpath='{.spec.host}' && echo

```

**Note:** The primary route is `data-science-gateway` in the `openshift-ingress` namespace. This provides access to all OpenShift AI workbenches and projects.

Once inside the dashboard, navigate to **Data Science Projects** → **hr-assistant** (or whatever you named your `${PROJECT}`).

![OpenShift AI Projects](docs/images/rhoai-1.png)

Inside the project you can see Workbenches. Open the **AnythingLLM** workbench.

![OpenShift AI Projects](docs/images/rhoai-2.png)

Finally, click on the **Assistant to the HR Representative** Workspace that's pre-created for you and you can start chatting with your assistant! 🎉

#### Direct Access URL

Your AnythingLLM workbench is accessible at:

```
https://<data-science-gateway-host>/notebook/hr-assistant/anythingllm/
```

To get the full URL:
```bash
echo "https://$(oc get route data-science-gateway -n openshift-ingress -o jsonpath='{.spec.host}')/notebook/${PROJECT}/anythingllm/"
```

**Note:** The route `data-science-gateway` in the `openshift-ingress` namespace handles all OpenShift AI workbench traffic and provides OAuth authentication.

#### Example Questions to Try:

```
Hi, one of our employees is going to get a raise, what do I need to keep in mind for this?
```

```
What are the key compliance considerations when handling employee misconduct in a bank?
```

```
How should I document a performance improvement plan for a regulated role?
```

The assistant will provide responses based on the seeded HR policy documents and citations.

![AnythingLLM](docs/images/anythingllm-1.png)

#### Performance Notes:

- **First response:** May take 20-30 seconds as the model processes the context
- **Subsequent responses:** ~15-25 seconds for typical answers (50-100 tokens)
- **Response quality:** OPT-125m provides good answers for factual questions and policy lookups
- **Limitations:** Being a smaller model (125M params), responses may be less sophisticated than larger models

#### Testing the API Directly:

You can also test the vLLM API directly:

```bash
# Port-forward to the vLLM service
oc port-forward -n ${PROJECT} svc/opt-125m-cpu-predictor 8080:80

# In another terminal, test the completions endpoint
curl -X POST "http://localhost:8080/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "opt-125m",
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 0.7
  }'

# Test the chat completions endpoint
curl -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "opt-125m",
    "messages": [
      {"role": "user", "content": "What is HR compliance?"}
    ],
    "max_tokens": 100
  }'
```



### Delete
```
helm uninstall ${PROJECT} --namespace ${PROJECT} 
```


### Switching Models

This deployment is designed to be flexible. To switch to a different model:

1. Update `helm/values.yaml`:
   ```yaml
   model:
     storageUri: "hf://facebook/opt-350m"  # or any HuggingFace model
     name: "opt-350m"
     maxModelLen: 2048
   ```

2. Ensure the model has a chat template, or update `helm/templates/vllm-chat-template-configmap.yaml`

3. Upgrade the deployment:
   ```bash
   helm upgrade ${PROJECT} helm/ --namespace ${PROJECT}
   ```

**Recommended CPU-friendly models:**
- `facebook/opt-125m` (current, fastest)
- `facebook/opt-350m` (better quality, slower)
- `facebook/opt-1.3b` (best quality, requires more resources)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (original model)

## Troubleshooting

### Workbench shows "Notebook image deleted"

**Cause:** The ImageStream `custom-anythingllm` doesn't exist in the `redhat-ods-applications` namespace.

**Solution:**
1. Check if ImageStream exists:
   ```bash
   oc get imagestream custom-anythingllm -n redhat-ods-applications
   ```

2. If missing, create it (see Prerequisites section above)

3. Delete and recreate the Notebook to pick up the ImageStream:
   ```bash
   oc delete notebook anythingllm -n hr-assistant
   # Wait for it to be recreated by Helm
   ```

### Workbench not accessible / "no healthy upstream"

**Cause:** The kube-rbac-proxy container may not have started, or services aren't created.

**Solution:**
1. Check pod has 3 containers running:
   ```bash
   oc get pod anythingllm-0 -n hr-assistant
   # Should show 3/3 Running
   ```

2. Check container names:
   ```bash
   oc get pod anythingllm-0 -n hr-assistant -o jsonpath='{.spec.containers[*].name}'
   # Should show: anythingllm anythingllm-automation kube-rbac-proxy
   ```

3. Check services were auto-created:
   ```bash
   oc get svc -n hr-assistant
   # Should show: anythingllm, anythingllm-kube-rbac-proxy
   ```

4. Check HTTPRoute backend:
   ```bash
   oc get httproute nb-hr-assistant-anythingllm -n redhat-ods-applications -o jsonpath='{.spec.rules[0].backendRefs[0]}'
   # Should point to: anythingllm-kube-rbac-proxy:8443
   ```

### vLLM pod not starting / model download fails

**Cause:** Network issues downloading from HuggingFace, or insufficient resources.

**Solution:**
1. Check pod logs:
   ```bash
   oc logs -n hr-assistant $(oc get pod -n hr-assistant -l app=isvc.opt-125m-cpu-predictor -o name) -c kserve-container
   ```

2. Verify network access to HuggingFace:
   ```bash
   oc debug -n hr-assistant $(oc get pod -n hr-assistant -l app=isvc.opt-125m-cpu-predictor -o name) -- curl -I https://huggingface.co
   ```

3. Check resource limits:
   ```bash
   oc describe pod -n hr-assistant $(oc get pod -n hr-assistant -l app=isvc.opt-125m-cpu-predictor -o name)
   # Look for "Insufficient memory" or "Insufficient cpu" events
   ```

### Storage issues

**Cause:** Storage class doesn't exist or PVC can't be provisioned.

**Solution:**
1. List available storage classes:
   ```bash
   oc get storageclass
   ```

2. Update `helm/values.yaml` with a valid storage class name

3. Check PVC status:
   ```bash
   oc get pvc -n hr-assistant
   ```

4. If PVC is pending, check events:
   ```bash
   oc describe pvc anythingllm -n hr-assistant
   ```

### References

**Model:**
- Model: [facebook/opt-125m on HuggingFace](https://huggingface.co/facebook/opt-125m)
- Paper: [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068)
- Model family: Meta's OPT (Open Pretrained Transformers)

**Runtime & Infrastructure:**
- Runtime built from: [vLLM CPU](https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html)
- Runtime image: [quay.io/rh-aiservices-bu/vllm-cpu-openai-ubi9](https://quay.io/repository/rh-aiservices-bu/vllm-cpu-openai-ubi9)
- Runtime code: [github.com/rh-aiservices-bu/llm-on-openshift](https://github.com/rh-aiservices-bu/llm-on-openshift/tree/main/serving-runtimes/vllm_runtime)
- AnythingLLM: [Mintplex-Labs/anything-llm](https://github.com/Mintplex-Labs/anything-llm)

**Fork Information:**
- Original repository: [rh-ai-quickstart/llm-cpu-serving](https://github.com/rh-ai-quickstart/llm-cpu-serving)
- This fork: [rocrisp/llm-cpu-serving](https://github.com/rocrisp/llm-cpu-serving)
- Changelog: [CHANGES.md](CHANGES.md)


## Tags

* **Industry:** Adopt and scale AI
* **Product:** OpenShift AI 
* **Use case:** Productivity
