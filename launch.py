import hydra

from omegaconf import DictConfig
import base64

from kubernetes import client, config
from kubejobs.jobs import KubernetesJob, KueueQueue


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


def send_message_command(launch_cfg: dict):
    # webhook - load from env
    config.load_kube_config()
    v1 = client.CoreV1Api()

    secret_name = launch_cfg["env_vars"]["SLACK_WEBHOOK"]["secret_name"]
    secret_key = launch_cfg["env_vars"]["SLACK_WEBHOOK"]["key"]

    secret = v1.read_namespaced_secret(secret_name, "informatics").data
    webhook = base64.b64decode(secret[secret_key]).decode("utf-8")
    return (
        """curl -X POST -H 'Content-type: application/json' --data '{"text":"Job started in '"$POD_NAME"'"}' """
        + webhook
        + " ; "
    )


def export_env_vars(launch_cfg: dict):
    cmd = ""
    for key in launch_cfg["env_vars"].keys():
        cmd += f" export {key}=${key} &&"
    cmd = cmd.strip(" &&") + " ; "
    return cmd


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    launch_cfg = dict(cfg)["launch"]
    job_name = launch_cfg["job_name"]
    is_completed = check_if_completed(job_name, namespace=launch_cfg["namespace"])

    if is_completed is True:
        print(f"Job '{job_name}' is completed. Launching a new job.")

        # TODO: make this interactive mode or not
        if launch_cfg["interactive"]:
            command = "while true; do sleep 60; done;"
        else:
            plan_craft_cfg = launch_cfg["plan_craft"]
            command = launch_cfg["command"]
            for key, value in plan_craft_cfg.items():
                command += f" ++.plancraft.{key}={value}"
            print(f"Command: {command}")

        # Create a Kubernetes Job with a name, container image, and command
        print(f"Creating job for: {command}")
        job = KubernetesJob(
            name=job_name,
            cpu_request=launch_cfg["cpu_request"],
            ram_request=launch_cfg["ram_request"],
            image="docker.io/gautierdag/plancraft:latest",
            gpu_type="nvidia.com/gpu",
            gpu_limit=launch_cfg["gpu_limit"],
            gpu_product=launch_cfg["gpu_product"],
            backoff_limit=0,
            command=["/bin/bash", "-c", "--"],
            args=[
                export_env_vars(launch_cfg) + send_message_command(launch_cfg) + command
            ],
            user_email="gautier.dagan@ed.ac.uk",
            namespace=launch_cfg["namespace"],
            kueue_queue_name=KueueQueue.INFORMATICS,
            secret_env_vars=launch_cfg["env_vars"],
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
