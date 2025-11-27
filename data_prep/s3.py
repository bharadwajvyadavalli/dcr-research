import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Optional, List
import logging
from pathlib import Path

# Suppress verbose S3 logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class S3Manager:
    """
    Manages AWS S3 operations for uploading and downloading files
    """

    def __init__(self, bucket_name: str, aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None, region_name: str = 'us-east-1'):
        """
        Initialize S3 Manager

        Args:
            bucket_name: Name of the S3 bucket
            aws_access_key_id: AWS access key (optional, can use env vars or AWS config)
            aws_secret_access_key: AWS secret key (optional, can use env vars or AWS config)
            region_name: AWS region (default: us-east-1)
        """
        self.bucket_name = bucket_name

        # Initialize S3 client
        session_kwargs = {'region_name': region_name}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs['aws_access_key_id'] = aws_access_key_id
            session_kwargs['aws_secret_access_key'] = aws_secret_access_key

        self.s3_client = boto3.client('s3', **session_kwargs)
        logger.info(f"S3 Manager initialized for bucket: {bucket_name}")

    def upload_file(self, local_path: str, s3_key: Optional[str] = None) -> bool:
        """
        Upload a file to S3

        Args:
            local_path: Path to local file
            s3_key: S3 object key (if None, uses the filename)

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(local_path):
            logger.error(f"File not found: {local_path}")
            return False

        if s3_key is None:
            s3_key = os.path.basename(local_path)

        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"✓ Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            return False
        except ClientError as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            return False

    def upload_directory(self, local_dir: str, s3_prefix: str = "") -> int:
        """
        Upload an entire directory to S3

        Args:
            local_dir: Path to local directory
            s3_prefix: Prefix for S3 keys (folder path in S3)

        Returns:
            Number of files successfully uploaded
        """
        if not os.path.isdir(local_dir):
            logger.error(f"Directory not found: {local_dir}")
            return 0

        uploaded_count = 0
        for root, dirs, files in os.walk(local_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")

                if self.upload_file(local_path, s3_key):
                    uploaded_count += 1

        logger.info(f"✓ Uploaded {uploaded_count} files from {local_dir}")
        return uploaded_count

    def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Download a file from S3

        Args:
            s3_key: S3 object key
            local_path: Local path to save file

        Returns:
            True if successful, False otherwise
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)

        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"✓ Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            return False
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.error(f"File not found in S3: {s3_key}")
            else:
                logger.error(f"Failed to download {s3_key}: {e}")
            return False

    def download_directory(self, s3_prefix: str, local_dir: str) -> int:
        """
        Download all files with a given prefix from S3

        Args:
            s3_prefix: Prefix for S3 keys (folder path in S3)
            local_dir: Local directory to save files

        Returns:
            Number of files successfully downloaded
        """
        try:
            # List all objects with the given prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix)

            downloaded_count = 0
            for page in pages:
                if 'Contents' not in page:
                    continue

                for obj in page['Contents']:
                    s3_key = obj['Key']
                    # Skip if it's a directory marker
                    if s3_key.endswith('/'):
                        continue

                    # Calculate local path
                    relative_path = os.path.relpath(s3_key, s3_prefix)
                    local_path = os.path.join(local_dir, relative_path)

                    if self.download_file(s3_key, local_path):
                        downloaded_count += 1

            logger.info(f"✓ Downloaded {downloaded_count} files to {local_dir}")
            return downloaded_count

        except ClientError as e:
            logger.error(f"Failed to list objects with prefix {s3_prefix}: {e}")
            return 0

    def list_files(self, prefix: str = "") -> List[str]:
        """
        List all files in the bucket with optional prefix

        Args:
            prefix: Optional prefix to filter files

        Returns:
            List of S3 keys
        """
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

            files = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if not obj['Key'].endswith('/'):
                            files.append(obj['Key'])

            logger.info(f"Found {len(files)} files with prefix '{prefix}'")
            return files

        except ClientError as e:
            logger.error(f"Failed to list files: {e}")
            return []

    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3

        Args:
            s3_key: S3 object key

        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"✓ Deleted s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete {s3_key}: {e}")
            return False

    def get_file_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate a presigned URL for temporary access to a file

        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds (default: 1 hour)

        Returns:
            Presigned URL or None if failed
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            logger.info(f"✓ Generated URL for {s3_key} (expires in {expiration}s)")
            return url
        except ClientError as e:
            logger.error(f"Failed to generate URL for {s3_key}: {e}")
            return None


def main():
    """Example usage"""
    # Initialize S3 manager (you can set credentials via env vars: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    s3_manager = S3Manager(
        bucket_name=os.getenv('S3_BUCKET_NAME', 'my-dcr-research-bucket'),
        region_name=os.getenv('AWS_REGION', 'us-east-1')
    )

    # Upload examples
    # s3_manager.upload_file('results.json', 'outputs/results.json')
    # s3_manager.upload_directory('cache', 'outputs/cache')

    # Download examples
    # s3_manager.download_file('outputs/results.json', 'downloaded_results.json')
    # s3_manager.download_directory('outputs', 'downloaded_outputs')

    # List files
    # files = s3_manager.list_files('outputs/')
    # print(f"Files in bucket: {files}")


if __name__ == "__main__":
    main()
