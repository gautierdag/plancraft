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

### Adding local environment variables

Create unique secret with environment variables:

```bash
kubectl create secret generic s2234411-hf --from-literal=HF_TOKEN=hf_***
kubectl create secret generic s2234411-openai --from-literal=OPENAI_API_KEY=sk-***
kubectl create secret generic s2234411-wandb --from-literal=WANDB_API_KEY=***
# Optional: slack webhook to get notified when pod starts
kubectl create secret generic s2234411-slack-webhook --from-literal=SLACK_WEBHOOK=***
```

### Update the launch config in your run config

Example of my config `configs/text-env/base.yaml` that declares the environment variables and GPU limits for the pod:

```yaml
launch:
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
    SLACK_WEBHOOK:
      secret_name: s2234411-slack-webhook
      key: SLACK_WEBHOOK
```

### Deploy the pod

To deply use `kubejobs` library, and the provided `launch.py` script.

```bash
python launch.py ++launch.job-name=gautier-test-job ++launch.gpu-type=NVIDIA-A100-SXM4-80GB ++launch.gpu-limit=1
```

#### Interactive session

To get an interactive session:

```bash
python launch.py ++launch.job-name=gautier-test-job ++launch.gpu-type=NVIDIA-A100-SXM4-40GB ++launch.gpu-limit=1 ++launch.interactive=True
```

Once the pod is live, you can run the following command:

```bash
kubectl exec -it <job-name> -- /bin/bash
```

#### Monitoring

To monitor the pod, you can use the following command:

```bash
kubectl logs -f <job-name>
```
