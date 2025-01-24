# Agents

## Setup

### Python version

please ensure you're using Python 3.11 or later. 

```
python3 --version
```

### Clone repo
```
git clone https://github.com/rahilvora/Agents.git
$ cd Agents
```

### Create an environment and install dependencies
#### Mac/Linux/WSL
```
$ python3 -m venv v-environment
$ source v-environment/bin/activate
$ pip install -r requirements.txt

### deploy the app
```
1. create deployment folder along side studio folder
2. create docker-compose.yml file in deployment folder. Make sure to update the image name to `article-agent` and API keys in the docker compose file.
3. copy python files from studio folder to deployment folder
4. copy langgraph.json file from studio folder to deployment folder
5. create docker image `langgraph build -t article-agent`
6. run docker compose `docker compose up`
```