#!/bin/bash

set -e

MINIO_ENDPOINT="http://localhost:9000"

if [ -z "$DR_LOCAL_S3_BUCKET" ]; then
    echo "Error: DR_LOCAL_S3_BUCKET environment variable is not set"
    echo "Source your system.env file: source thunder_files/MODEL_ID/system.env"
    exit 1
fi

BUCKET_NAME="$DR_LOCAL_S3_BUCKET"
echo "Creating bucket: $BUCKET_NAME"

if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed"
    exit 1
fi

if ! curl -f -s "$MINIO_ENDPOINT/minio/health/live" > /dev/null 2>&1; then
    echo "Error: Cannot connect to MinIO at $MINIO_ENDPOINT"
    echo "Make sure MinIO is running"
    exit 1
fi

if ! aws configure list &> /dev/null; then
    export AWS_ACCESS_KEY_ID="minioadmin"
    export AWS_SECRET_ACCESS_KEY="minioadmin"
fi

if aws s3api head-bucket --bucket "$BUCKET_NAME" --endpoint-url "$MINIO_ENDPOINT" 2>/dev/null --profile minio; then
    echo "Bucket already exists: $BUCKET_NAME"
else
    aws s3api create-bucket --bucket "$BUCKET_NAME" --endpoint-url "$MINIO_ENDPOINT" --profile minio

    POLICY_FILE=$(mktemp)
    trap "rm -f $POLICY_FILE" EXIT

    cat > "$POLICY_FILE" << EOF
{
    "Version": "2012-10-17",
    "Id": "AwsDeepracerServiceAccess",
    "Statement": [
        {
            "Sid": "Stmt1753912486517",
            "Effect": "Allow",
            "Principal": {
                "Service": "deepracer.amazonaws.com"
            },
            "Action": [
                "s3:GetObjectAcl",
                "s3:GetObject",
                "s3:PutObject",
                "s3:PutObjectAcl",
                "s3:ListBucket",
                "s3:ListBucketVersions"
            ],
            "Resource": [
                "arn:aws:s3:::$BUCKET_NAME",
                "arn:aws:s3:::$BUCKET_NAME/*"
            ]
        }
    ]
}
EOF

    aws s3api put-bucket-policy --bucket "$BUCKET_NAME" --policy file://"$POLICY_FILE" --profile minio --endpoint-url "$MINIO_ENDPOINT" 2>/dev/null || true
fi

if [ -n "$1" ]; then
    LOCAL_DIR="$1"
    S3_PREFIX="${2:-}"

    if [ ! -d "$LOCAL_DIR" ]; then
        echo "Error: Local directory '$LOCAL_DIR' does not exist"
        exit 1
    fi

    echo "Uploading from local directory: $LOCAL_DIR"
    if [ -n "$S3_PREFIX" ]; then
        echo "Uploading to S3 prefix: $S3_PREFIX"
        aws s3 sync "$LOCAL_DIR" "s3://$BUCKET_NAME/$S3_PREFIX" --profile minio --endpoint-url "$MINIO_ENDPOINT"
    else
        echo "Uploading to bucket root"
        aws s3 sync "$LOCAL_DIR" "s3://$BUCKET_NAME" --profile minio --endpoint-url "$MINIO_ENDPOINT"
    fi

    echo "Upload completed successfully!"

    echo "Files in bucket:"
    aws s3 ls "s3://$BUCKET_NAME" --profile minio --endpoint-url "$MINIO_ENDPOINT" --recursive
else
    echo "Bucket created successfully: $BUCKET_NAME"
    echo ""
    echo "Usage for uploading:"
    echo "$0 <local_directory> [s3_prefix]"
    echo ""
    echo "Examples:"
    echo "$0 ./thunder_files/MODEL_ID custom_files"
    echo "$0 ./my_model_files rl-deepracer-sagemaker"
fi
