docker build --progress=plain -t pseugc-app .
docker tag pseugc-app sksdotsauravs/pseugc-app:latest
docker login
docker push sksdotsauravs/pseugc-app:latest

kubectl -n s81481 apply -f .\ollama-pod.yml
kubectl -n s81481 port-forward ollama-pod 11434:11434
kubectl -n s81481 delete pods ollama-pod --grace-period=0 --force
kubectl -n s81481 delete pvc s81481-ollama-pvc