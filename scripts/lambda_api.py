import os
import click
import requests
import yaml


LAMBDA_API_ROOT = "https://cloud.lambda.ai/api/v1"


class LambdaAPI:

  def __init__(self, api_key=None):
    self._api_root = LAMBDA_API_ROOT
    self._api_key = api_key or os.getenv("LAMBDA_API_KEY")
    if self._api_key is None:
      raise ValueError("No environment variable found for LAMBDA_API_KEY")

  def get(self, endpoint, params=None):
    url = f"{self._api_root}/{endpoint}"
    headers = {
      "Authorization": f"Bearer {self._api_key}"
    }
    response = requests.get(url, headers=headers, params=params)
    return response
  
  def post(self, endpoint, data=None):
    url = f"{self._api_root}/{endpoint}"
    headers = {
      "Authorization": f"Bearer {self._api_key}"
    }
    response = requests.post(url, headers=headers, json=data)
    return response

  def list_instances(self):
    return self.get("instances")
  
  def list_instance_types(self, region=None):
    response = self.get("instance-types")
    if region is None:
      return response.json()
    
    results = {}
    for type_name, type_data in response.json()["data"].items():
      regions = [r["name"] for r in type_data.get("regions_with_capacity_available", [])]
      if region in regions:
        results[type_name] = type_data
    return {"data": results}

  def create_instance(self, spec_filepath):
    with open(spec_filepath, "r") as f:
      spec = yaml.safe_load(f)
    print(spec)
    return self.post("instance-operations/launch", data=spec)
  
  def terminate_instance(self, instance_id=None):
    if instance_id is None:
      response = self.list_instances()
      instance_data = response.json()["data"]
      if not instance_data:
        raise ValueError("No instances are running")
      instance_id = instance_data[0].get("id")

    if click.confirm(f"About to delete instance {instance_id}?", default=False):
      post_data = {
        "instance_ids": [ instance_id ]
      }
      response = self.post("instance-operations/terminate", data=post_data)
      return response


@click.group()
def cli():
  pass


@cli.command()
@click.argument("config")
def launch_instance(config):
  api = LambdaAPI()
  response = api.create_instance(config)
  print(response.text)


@cli.command()
def list_instances():
  api = LambdaAPI()
  response = api.list_instances()
  print(response.text)


@cli.command()
@click.argument("region", default="us-east-1")
def list_instance_types(region):
  api = LambdaAPI()
  response = api.list_instance_types(region)
  print(response["data"].keys())


@cli.command()
@click.argument("instance_id", required=False)
def terminate_instance(instance_id):
  api = LambdaAPI()
  api.terminate_instance(instance_id=instance_id)


if __name__ == "__main__":
  cli()