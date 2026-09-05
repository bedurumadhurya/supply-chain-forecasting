#!/bin/bash

# AWS Deployment Script for Supply Chain Forecasting System
# This script automates deployment to AWS infrastructure

set -e  # Exit on error

echo "=========================================="
echo "AWS Deployment Script"
echo "Supply Chain Demand Forecasting System"
echo "=========================================="

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI not found. Please install it first."
    exit 1
fi

# Check AWS credentials
echo "Checking AWS credentials..."
aws sts get-caller-identity > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ AWS credentials configured"
else
    echo "Error: AWS credentials not configured"
    echo "Please run: aws configure"
    exit 1
fi

# Variables
AWS_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET_NAME:-supply-chain-forecasting-bucket}"
ECR_REPO="supply-chain-forecasting"
APP_NAME="forecasting-api"

echo ""
echo "Configuration:"
echo "  AWS Region: $AWS_REGION"
echo "  S3 Bucket: $S3_BUCKET"
echo "  ECR Repository: $ECR_REPO"
echo ""

# Step 1: Create S3 bucket for data and models
echo "Step 1: Setting up S3 bucket..."
if aws s3 ls "s3://$S3_BUCKET" 2>&1 | grep -q 'NoSuchBucket'; then
    echo "Creating S3 bucket: $S3_BUCKET"
    aws s3 mb "s3://$S3_BUCKET" --region "$AWS_REGION"
    echo "✓ S3 bucket created"
else
    echo "✓ S3 bucket already exists"
fi

# Create folder structure in S3
echo "Creating S3 folder structure..."
aws s3api put-object --bucket "$S3_BUCKET" --key data/raw/ --region "$AWS_REGION"
aws s3api put-object --bucket "$S3_BUCKET" --key data/processed/ --region "$AWS_REGION"
aws s3api put-object --bucket "$S3_BUCKET" --key models/ --region "$AWS_REGION"
aws s3api put-object --bucket "$S3_BUCKET" --key forecasts/ --region "$AWS_REGION"
echo "✓ S3 folder structure created"

# Step 2: Upload models to S3 (if they exist)
echo ""
echo "Step 2: Uploading models to S3..."
if [ -d "models" ] && [ "$(ls -A models/*.pkl 2>/dev/null)" ]; then
    aws s3 sync models/ "s3://$S3_BUCKET/models/" --exclude "*.gitkeep"
    echo "✓ Models uploaded to S3"
else
    echo "⚠ No models found in models/ directory"
fi

# Step 3: Create ECR repository and push Docker image
echo ""
echo "Step 3: Building and pushing Docker image..."

# Create ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Creating ECR repository: $ECR_REPO"
    aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION"
    echo "✓ ECR repository created"
else
    echo "✓ ECR repository already exists"
fi

# Get ECR login
echo "Logging into ECR..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com"

# Build Docker image
echo "Building Docker image..."
docker build -t "$ECR_REPO:latest" .

# Tag and push image
ECR_URI="$(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest"
docker tag "$ECR_REPO:latest" "$ECR_URI"
echo "Pushing image to ECR..."
docker push "$ECR_URI"
echo "✓ Docker image pushed to ECR"

# Step 4: Create CloudWatch Log Group
echo ""
echo "Step 4: Setting up CloudWatch logging..."
aws logs create-log-group --log-group-name "/aws/forecasting/$APP_NAME" --region "$AWS_REGION" 2>/dev/null || true
echo "✓ CloudWatch log group configured"

# Deployment instructions
echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Deploy to EC2 or ECS using the Docker image:"
echo "   ECR Image URI: $ECR_URI"
echo ""
echo "2. For EC2 deployment:"
echo "   - Launch an EC2 instance (t3.medium recommended)"
echo "   - Install Docker"
echo "   - Pull and run the container:"
echo "     aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URI"
echo "     docker pull $ECR_URI"
echo "     docker run -d -p 80:8000 --name forecasting-api $ECR_URI"
echo ""
echo "3. Configure security groups to allow inbound traffic on port 80"
echo ""
echo "4. Access the API at: http://<EC2-PUBLIC-IP>/health"
echo ""
echo "5. Monitor logs in CloudWatch: /aws/forecasting/$APP_NAME"
echo ""
