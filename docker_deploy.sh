# Build container
docker build . --network=host --tag streamlitapp

# artifactory: europe-west3-docker.pkg.dev/medtech-hack24cop-4047/lenns-artifactory/
# Tag container
docker tag streamlitapp:latest europe-west3-docker.pkg.dev/medtech-hack24cop-4047/lenns-artifactory/streamlitapp:latest

# Push to artifactory
docker push europe-west3-docker.pkg.dev/medtech-hack24cop-4047/lenns-artifactory/streamlitapp:latest

# Create new revision
gcloud run deploy streamlitapp --image europe-west3-docker.pkg.dev/medtech-hack24cop-4047/lenns-artifactory/streamlitapp:latest --platform managed --region europe-west3 --allow-unauthenticated --memory 4Gi --min-instances 1 --cpu 2 


