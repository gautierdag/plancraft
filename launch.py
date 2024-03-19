#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

sys.path.append(os.getcwd())

import yaml

from kubernetes import client, config
from kubejobs.jobs import KubernetesJob, KueueQueue


def argument_parser():
    parser = argparse.ArgumentParser(description="Backend Runner")
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="launch.yaml"
    )
    parser.add_argument("--job-name", "-n", type=str, default="gautier-test-job")
    parser.add_argument("--gpu-type", type=str, default=None)
    parser.add_argument("--gpu-limit", type=int, default=None)
    parser.add_argument("--interactive", type=str, default=True)
    parser.add_argument("--namespace", type=str, default="informatics")
    args = parser.parse_args()
    return args


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


def send_message_command():
    return """curl -X POST -H 'Content-type: application/json' --data '{"text":"Job launched!"}' https://hooks.slack.com/services/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;"""


def export_env_vars():
    return """export HF_TOKEN=$HF_TOKEN && export OPENAI_API_KEY=$OPENAI_API_KEY && export WANDB_API_KEY=$WANDB_API_KEY;"""


def main():
    args = argument_parser()
    configs = yaml.safe_load(open(args.config, "r"))

    job_name = args.job_name
    is_completed = check_if_completed(job_name, namespace=args.namespace)

    if is_completed is True:
        print(f"Job '{job_name}' is completed. Launching a new job.")

        # TODO: make this interactive mode
        command = "while true; do sleep 60; done;"

        secret_env_vars = configs["env_vars"]

        # Create a Kubernetes Job with a name, container image, and command
        print(f"Creating job for: {command}")
        job = KubernetesJob(
            name=job_name,
            cpu_request="8",
            ram_request="80Gi",
            image="docker.io/gautierdag/plancraft:latest",
            gpu_type="nvidia.com/gpu",
            gpu_limit=configs["gpu_limit"]
            if args.gpu_limit is None
            else args.gpu_limit,
            gpu_product=configs["gpu_product"]
            if args.gpu_type is None
            else args.gpu_type,
            backoff_limit=1,
            command=["/bin/bash", "-c", "--"],
            args=[export_env_vars() + send_message_command() + command],
            user_email="gautier.dagan@ed.ac.uk",
            namespace=args.namespace,
            kueue_queue_name=KueueQueue.INFORMATICS,
            secret_env_vars=secret_env_vars,
        )

        job_yaml = job.generate_yaml()
        print(job_yaml)

        # Run the Job on the Kubernetes cluster
        job.run()


if __name__ == "__main__":
    main()
