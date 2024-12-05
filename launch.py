import hydra

from omegaconf import DictConfig
import base64

from kubernetes import client, config
from kubejobs.jobs import KubernetesJob, KueueQueue

from plancraft.config import EvalConfig


def check_if_completed(job_name: str, namespace: str = "informatics") -> bool:
    # Load the kube config
    config.load_kube_config()

    # Create an instance of the API class
    api = client.BatchV1Api()

    job_exists = False
    is_completed = True

    # Check if the job exists in the specified namespace
    jobs = api.list_namespaced_job(namespace)
    if job_name in {job.metadata.name for job in jobs.items}:
        job_exists = True

    if job_exists is True:
        job = api.read_namespaced_job(job_name, namespace)
        is_completed = False

        # Check the status conditions
        if job.status.conditions:
            for condition in job.status.conditions:
                if condition.type == "Complete" and condition.status == "True":
                    is_completed = True
                elif condition.type == "Failed" and condition.status == "True":
                    print(f"Job {job_name} has failed.")
        else:
            print(f"Job {job_name} still running or status is unknown.")

        if is_completed:
            api_res = api.delete_namespaced_job(
                name=job_name,
                namespace=namespace,
                body=client.V1DeleteOptions(propagation_policy="Foreground"),
            )
            print(f"Job '{job_name}' deleted. Status: {api_res.status}")
    return is_completed


def send_message_command(cfg: EvalConfig):
    # webhook - load from env
    config.load_kube_config()
    v1 = client.CoreV1Api()

    secret_name = cfg.launch.env_vars["SLACK_WEBHOOK"]["secret_name"]
    secret_key = cfg.launch.env_vars["SLACK_WEBHOOK"]["key"]

    secret = v1.read_namespaced_secret(secret_name, "informatics").data
    webhook = base64.b64decode(secret[secret_key]).decode("utf-8")
    return (
        """curl -X POST -H 'Content-type: application/json' --data '{"text":"Job started in '"$POD_NAME"'"}' """
        + webhook
        + " ; "
    )


def export_env_vars(cfg: EvalConfig):
    cmd = ""
    for key in cfg.launch.env_vars.keys():
        cmd += f" export {key}=${key} &&"
    cmd = cmd.strip(" &&") + " ; "
    return cmd


def flatten_cfg(cfg):
    # for some reason hydra wraps file paths from config path
    if len(cfg) == 1:
        return flatten_cfg(cfg[list(cfg.keys())[0]])
    return cfg


@hydra.main(config_path="configs", config_name="evals/llama8B", version_base=None)
def main(cfg: DictConfig):
    cfg = EvalConfig(**flatten_cfg(dict(cfg)))
    job_name = cfg.launch.job_name
    is_completed = check_if_completed(job_name, namespace=cfg.launch.namespace)

    if is_completed is True:
        print(f"Job '{job_name}' is completed. Launching a new job.")

        # TODO: make this interactive mode or not
        if cfg.launch.interactive:
            command = "while true; do sleep 60; done;"
        else:
            plancraft_cfg = dict(cfg)["plancraft"]
            command = cfg.launch.command
            for key, value in plancraft_cfg.items():
                command += f" ++plancraft.{key}={value}"
            print(f"Command: {command}")

        # Create a Kubernetes Job with a name, container image, and command
        print(f"Creating job for: {command}")
        job = KubernetesJob(
            name=job_name,
            cpu_request=cfg.launch.cpu_request,
            ram_request=cfg.launch.ram_request,
            image="docker.io/gautierdag/plancraft:latest",
            gpu_type="nvidia.com/gpu",
            gpu_limit=cfg.launch.gpu_limit,
            gpu_product=cfg.launch.gpu_product,
            backoff_limit=0,
            command=["/bin/bash", "-c", "--"],
            args=[export_env_vars(cfg) + send_message_command(cfg) + command],
            user_email="gautier.dagan@ed.ac.uk",
            namespace=cfg.launch.namespace,
            kueue_queue_name=KueueQueue.INFORMATICS,
            secret_env_vars=cfg.launch.env_vars,
            volume_mounts={
                "nfs": {"mountPath": "/nfs", "server": "10.24.1.255", "path": "/"}
            },
        )

        job_yaml = job.generate_yaml()
        print(job_yaml)

        # Run the Job on the Kubernetes cluster
        job.run()
    else:
        print(f"Job '{job_name}' is still running.")


if __name__ == "__main__":
    main()
