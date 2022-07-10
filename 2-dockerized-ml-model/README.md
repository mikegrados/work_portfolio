## Dockerized machine learning model

Docker container with a logistic regression model for the Titanic dataset. Good practices of writing code have been followed with the help of the libraries pycodestyle and pyment.

___

- Build image
  > docker build -t **image-name** -f Dockerfile .

- Run in a container the file "inference.py"
  > docker run **image-name** python3 /src/logreg/inference.py