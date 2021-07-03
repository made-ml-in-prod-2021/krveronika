## 1. Развернуть кластер Kubernetes
Кластер поднят локально с помощью https://kind.sigs.k8s.io/docs/user/quick-start/

- запущен докер
- kind create cluster
- проверка: kubectl cluster-info --context kind-kind

## 2. Простой pod manifest
- Deploy: 
  
kubectl apply -f kubernetes_manifests/online-inference-pod.yaml

- Проверка: 
  
kubectl get pods

2а) Прописала requests/limits: requests - нужны, чтобы Kubernetes определил исходя из доступных ресурсов, на какой хост поместить pod. limits - отвечает за пороговые максимальные значения, в случае превышения этих значений k8s начнет ограничивать или перезапускать контейнер.

## 3. Liveness & Readiness