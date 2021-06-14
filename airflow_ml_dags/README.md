# Homework #3 "ML in production", MADE-spring-2021.

## Настройка окружения и деплой Airflow
1. Окружение (переменные + credentials)
```
python3 -m venv venv_air
source venv_air/bin/activate
pip install -e .

chmod +x bin/scripts/airflow_setup.sh
source bin/scripts/airflow_setup.sh

deactivate
```
2. Билдинг базового образа
```
docker build ./images/airflow-ml-base -t airflow-ml-base:latest
```
3. Запустить Airflow
```
docker compose up --build
```  
 - в браузере  http://localhost:8090/
 - остановить Airflow
 ```
docker compose down
``` 

1.5. Тестирование  
```pytest -v ```  
