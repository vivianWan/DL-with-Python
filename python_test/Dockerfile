FROM python:3.6.6-jessie

# Install python packages
RUN pip install --no-cache-dir pymongo[tls,srv]==3.6.1 pprint

# Configure the Environment Variable with ENV
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Setup the working folder
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy the project files into the docker image (add files to the .dockerignore file in order to exlcude them from this copy)
COPY . .

CMD python test.py