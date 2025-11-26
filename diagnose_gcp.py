"""Diagnostic script to check GCP configuration."""

import os
import json
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("GCP CONFIGURATION DIAGNOSTIC")
print("=" * 60)

# Check 1: Environment variables
print("\n1. ENVIRONMENT VARIABLES:")
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

print(f"GOOGLE_CLOUD_PROJECT: {project_id}")
print(f"GOOGLE_APPLICATION_CREDENTIALS: {creds_path}")

if not project_id:
    print("ERROR: GOOGLE_CLOUD_PROJECT not set!")
else:
    print("Project ID is set")

if not creds_path:
    print("ERROR: GOOGLE_APPLICATION_CREDENTIALS not set!")
else:
    print("Credentials path is set")

# Check 2: Service account file exists
print("\n2. SERVICE ACCOUNT FILE:")
if creds_path and os.path.exists(creds_path):
    print(f"File exists at: {creds_path}")

    # Try to read the file
    try:
        with open(creds_path, "r") as f:
            creds_data = json.load(f)
        print(f"File is valid JSON")
        print(f"Service account email: {creds_data.get('client_email')}")
        print(f"Project ID in file: {creds_data.get('project_id')}")

        # Check if project IDs match
        if creds_data.get("project_id") != project_id:
            print(f"WARNING: Project ID mismatch!")
            print(f".env has: {project_id}")
            print(f"JSON has: {creds_data.get('project_id')}")
    except Exception as e:
        print(f"ERROR reading file: {e}")
else:
    print(f"ERROR: File not found at {creds_path}")

# Check 3: Try to connect to Vertex AI
print("\n3. VERTEX AI CONNECTION:")
try:
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location="us-central1")
    print("Successfully initialized Vertex AI")
    print(f"Using project: {project_id}")
    print(f"Using location: us-central1")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
