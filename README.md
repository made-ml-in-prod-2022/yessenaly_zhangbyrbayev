# yessenaly_zhangbyrbayev
ML in prod homeworks
Команды для поднятия докер имеджа и его запуска:
docker build --tag online_inference .
docker run --publish 8000:80 online_inference
