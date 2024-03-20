# How to use with Kubernetes cluster

## Building image with kaniko

Ensure you have a `Dockerfile` at the root of your repository.

### Register a secret with docker-hub credentials

Save a file as `config.json` with the following:

```bash
{
    "auths": {
        "https://index.docker.io/v1/": {
            "auth": "base64 encode of username:password"
        }
    }
}
```

Then run the following command:

```bash
kubectl create secret generic regcred-XXX --from-file=config.json=config.json --namespace=your-namespace
```

Replace `XXX` with your username and `your-namespace` with your namespace.

### Build Image on Kube

Create a file `kaniko.yaml` with the following:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kaniko
  labels:
    eidf/user: "s2234411" # Replace this with your EIDF username
spec:
  containers:
  - name: kaniko
    resources:
      requests:
        cpu: "500m"  # Requests 0.5 CPU cores
        memory: "1Gi"  # Requests 1 GiB of memory
      limits:
        cpu: "1"  # Limits to 1 CPU core
        memory: "2Gi"  # Limits to 2 GiB of memory
    image: gcr.io/kaniko-project/executor:latest
    args: ["--dockerfile=Dockerfile",
           "--context=git://github.com/gautierdag/plancraft.git#main", # Replace with your git repo - must be public
           "--destination=docker.io/gautierdag/plancraft:latest", # Replace with your docker hub image
           "--cache=true"]
    volumeMounts:
      - name: docker-config
        mountPath: /kaniko/.docker
  volumes:
    - name: docker-config
      secret:
        secretName: regcred-XXX
  restartPolicy: Never
```

Replace `s2234411` with your EIDF username, `gautierdag/plancraft` with your docker hub image and `XXX` with your username.

Then run the following command:

```bash
kubectl apply -f build/kaniko-pod.yaml
```

This will create a pod that will build your image and push it to docker hub.

## Using the image

Once the image is built and pushed to the hub, you can use it in your kubernetes cluster.

### Adding local environment variables

Create unique secret with environment variables:

```bash
kubectl create secret generic s2234411-hf --from-literal=HF_TOKEN=hf_***
kubectl create secret generic s2234411-openai --from-literal=OPENAI_API_KEY=sk-***
kubectl create secret generic s2234411-wandb --from-literal=WANDB_API_KEY=***
```

### Update the launch.yaml

Example of my `launch.yaml` that declares the environment variables and GPU limits for the pod:

```yaml
gpu_limit: 1
gpu_product: NVIDIA-A100-SXM4-80GB
env_vars:
  HF_TOKEN:
    secret_name: s2234411-hf
    key: HF_TOKEN
  OPENAI_API_KEY:
    secret_name: s2234411-openai
    key: OPENAI_API_KEY
  WANDB_API_KEY:
    secret_name: s2234411-wandb
    key: WANDB_API_KEY
```

### Deploy the pod

To deply use `kubejobs` library, and the provided `launch.py` script.

```bash
python launch.py --config launch.yaml --job-name gautier-test-job --gpu-type NVIDIA-A100-SXM4-80GB --gpu-limit 1  --namespace informatics
```

#### Interactive session

To get an interactive session with the pod, you can run the following command:

```bash
kubectl exec -it <job-name> -- /bin/bash
```
