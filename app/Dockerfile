FROM tiangolo/uwsgi-nginx-flask
#ENV STATIC_URL /static
#ENV STATIC_PATH /var/www/app/static
#COPY ./requirements.txt /var/www/requirements.txt
COPY . /app
WORKDIR /app
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["main.py"]

