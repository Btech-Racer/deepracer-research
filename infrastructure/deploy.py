#!/usr/bin/env python3

import time
from pathlib import Path
from typing import Optional

import boto3
import typer
from botocore.exceptions import ClientError, NoCredentialsError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(help="Deploy AWS DeepRacer Research Infrastructure")
console = Console()


def validate_aws_credentials() -> bool:
    """Validate AWS credentials are configured."""
    try:
        sts = boto3.client("sts")
        sts.get_caller_identity()
        return True
    except NoCredentialsError:
        console.print(" AWS credentials not configured", style="red")
        console.print("Configure with: aws configure", style="yellow")
        return False
    except Exception as e:
        console.print(f" Error validating credentials: {e}", style="red")
        return False


def get_stack_status(cf_client, stack_name: str) -> Optional[str]:
    """Get CloudFormation stack status."""
    try:
        response = cf_client.describe_stacks(StackName=stack_name)
        return response["Stacks"][0]["StackStatus"]
    except ClientError as e:
        if "does not exist" in str(e):
            return None
        raise


def wait_for_stack_operation(cf_client, stack_name: str, operation: str) -> bool:
    """Wait for CloudFormation stack operation to complete."""
    success_statuses = {"CREATE": ["CREATE_COMPLETE"], "UPDATE": ["UPDATE_COMPLETE"], "DELETE": ["DELETE_COMPLETE"]}

    failed_statuses = {
        "CREATE": ["CREATE_FAILED", "ROLLBACK_COMPLETE", "ROLLBACK_FAILED"],
        "UPDATE": ["UPDATE_FAILED", "UPDATE_ROLLBACK_COMPLETE", "UPDATE_ROLLBACK_FAILED"],
        "DELETE": ["DELETE_FAILED"],
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]{operation.title()}ing stack {stack_name}...", total=None)

        while True:
            status = get_stack_status(cf_client, stack_name)

            if not status and operation == "DELETE":
                progress.update(task, description=f"[green]Stack {stack_name} deleted successfully")
                return True

            if status in success_statuses.get(operation, []):
                progress.update(task, description=f"[green]Stack {stack_name} {operation.lower()}d successfully")
                return True

            if status in failed_statuses.get(operation, []):
                progress.update(task, description=f"[red]Stack {stack_name} {operation.lower()} failed: {status}")
                return False

            time.sleep(10)


def load_template(template_path: Path) -> str:
    """Load CloudFormation template."""
    if not template_path.exists():
        console.print(f" Template file not found: {template_path}", style="red")
        raise typer.Exit(1)

    return template_path.read_text()


def deploy_stack(cf_client, stack_name: str, template_body: str, parameters: dict) -> bool:
    """Deploy CloudFormation stack."""
    cf_parameters = [{"ParameterKey": key, "ParameterValue": value} for key, value in parameters.items()]

    current_status = get_stack_status(cf_client, stack_name)

    try:
        if current_status is None:
            console.print(f"Creating new stack: {stack_name}", style="cyan")
            cf_client.create_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=cf_parameters,
                Capabilities=["CAPABILITY_NAMED_IAM"],
                Tags=[
                    {"Key": "Purpose", "Value": "DeepRacer Research"},
                    {"Key": "ManagedBy", "Value": "CloudFormation"},
                    {"Key": "CreatedBy", "Value": "deployment-script"},
                ],
            )
            return wait_for_stack_operation(cf_client, stack_name, "CREATE")
        else:
            console.print(f"Updating existing stack: {stack_name}", style="cyan")
            try:
                cf_client.update_stack(
                    StackName=stack_name,
                    TemplateBody=template_body,
                    Parameters=cf_parameters,
                    Capabilities=["CAPABILITY_NAMED_IAM"],
                )
                return wait_for_stack_operation(cf_client, stack_name, "UPDATE")
            except ClientError as e:
                if "No updates are to be performed" in str(e):
                    console.print("No changes detected - stack is up to date", style="green")
                    return True
                raise

    except ClientError as e:
        console.print(f" Deployment failed: {e}", style="red")
        return False


def get_stack_outputs(cf_client, stack_name: str) -> dict:
    """Get CloudFormation stack outputs."""
    try:
        response = cf_client.describe_stacks(StackName=stack_name)
        outputs = response["Stacks"][0].get("Outputs", [])
        return {output["OutputKey"]: output["OutputValue"] for output in outputs}
    except ClientError:
        return {}


def display_outputs(outputs: dict):
    """Display stack outputs in a formatted table."""
    if not outputs:
        console.print("No stack outputs available", style="yellow")
        return

    table = Table(title="Stack Outputs", show_header=True, header_style="bold magenta")
    table.add_column("Output", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    for key, value in outputs.items():
        table.add_row(key, value)

    console.print(table)


@app.command()
def deploy(
    project_name: str = typer.Option("deepracer-research", "--project", "-p", help="Project name prefix"),
    environment: str = typer.Option("research", "--env", "-e", help="Environment type"),
    bucket_suffix: str = typer.Option("thesis-data", "--suffix", "-s", help="S3 bucket suffix"),
    region: str = typer.Option("us-east-1", "--region", "-r", help="AWS region"),
    stack_name: Optional[str] = typer.Option(None, "--stack-name", help="Override stack name"),
):
    """Deploy the AWS DeepRacer research infrastructure."""

    if not validate_aws_credentials():
        raise typer.Exit(1)

    boto3.setup_default_session(region_name=region)
    cf_client = boto3.client("cloudformation", region_name=region)

    if not stack_name:
        stack_name = f"{project_name}-infrastructure"

    template_path = Path(__file__).parent / "cloudformation-template.yaml"
    template_body = load_template(template_path)

    parameters = {"ProjectName": project_name, "Environment": environment, "S3BucketSuffix": bucket_suffix}

    console.print(f"Deployment Configuration:", style="bold")
    console.print(f"  Stack Name: {stack_name}")
    console.print(f"  Region: {region}")
    console.print(f"  Project: {project_name}")
    console.print(f"  Environment: {environment}")
    console.print(f"  Bucket Suffix: {bucket_suffix}")
    console.print()

    if deploy_stack(cf_client, stack_name, template_body, parameters):
        console.print("Deployment completed successfully!", style="green bold")

        outputs = get_stack_outputs(cf_client, stack_name)
        display_outputs(outputs)
    else:
        console.print(" Deployment failed!", style="red bold")
        raise typer.Exit(1)


@app.command()
def delete(
    stack_name: str = typer.Argument(..., help="Stack name to delete"),
    region: str = typer.Option("us-east-1", "--region", "-r", help="AWS region"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete the AWS DeepRacer infrastructure stack."""

    if not validate_aws_credentials():
        raise typer.Exit(1)

    cf_client = boto3.client("cloudformation", region_name=region)

    if get_stack_status(cf_client, stack_name) is None:
        console.print(f" Stack '{stack_name}' does not exist", style="red")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Are you sure you want to delete stack '{stack_name}'?")
        if not confirm:
            console.print("Deletion cancelled", style="yellow")
            raise typer.Exit(0)

    try:
        console.print(f"Deleting stack: {stack_name}", style="cyan")
        cf_client.delete_stack(StackName=stack_name)

        if wait_for_stack_operation(cf_client, stack_name, "DELETE"):
            console.print("Stack deleted successfully!", style="green bold")
        else:
            console.print(" Stack deletion failed!", style="red bold")
            raise typer.Exit(1)

    except ClientError as e:
        console.print(f" Deletion failed: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def status(
    stack_name: str = typer.Argument(..., help="Stack name to check"),
    region: str = typer.Option("us-east-1", "--region", "-r", help="AWS region"),
):
    """Check the status of the infrastructure stack."""

    if not validate_aws_credentials():
        raise typer.Exit(1)

    cf_client = boto3.client("cloudformation", region_name=region)

    current_status = get_stack_status(cf_client, stack_name)

    if current_status is None:
        console.print(f"Stack '{stack_name}' does not exist", style="yellow")
    else:
        console.print(f"Stack Status: {current_status}", style="cyan")

        if "COMPLETE" in current_status:
            outputs = get_stack_outputs(cf_client, stack_name)
            display_outputs(outputs)


@app.command()
def validate(region: str = typer.Option("us-east-1", "--region", "-r", help="AWS region")):
    """Validate the CloudFormation template."""

    if not validate_aws_credentials():
        raise typer.Exit(1)

    template_path = Path(__file__).parent / "cloudformation-template.yaml"
    template_body = load_template(template_path)

    cf_client = boto3.client("cloudformation", region_name=region)

    try:
        console.print("Validating CloudFormation template...", style="cyan")
        cf_client.validate_template(TemplateBody=template_body)
        console.print("Template is valid!", style="green bold")

    except ClientError as e:
        console.print(f" Template validation failed: {e}", style="red")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
