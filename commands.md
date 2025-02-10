docker build --progress=plain -t pseugc-app .
docker tag pseugc-app sksdotsauravs/pseugc-app:latest
docker login
docker push sksdotsauravs/pseugc-app:latest