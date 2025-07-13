# Python Flask Application

A simple Python Flask web application that can be deployed using devfiles.

**Note:** The application uses port **8081** for HTTP traffic.

## Project Structure

This project includes:

- `app.py` - Main Flask application
- `devfile.yaml` - Devfile configuration for containerization and deployment
- `docker/Dockerfile` - Docker configuration for building the container image
- `deploy.yaml` - Kubernetes deployment configuration
- `requirements.txt` - Python dependencies

## How it works

1. The `devfile.yaml` file contains an `image-build` component that builds a Docker image using the `docker/Dockerfile`
2. The `docker/Dockerfile` contains instructions to build the Flask application as a container image
3. The `devfile.yaml` `kubernetes-deploy` component deploys the built container image using the `deploy.yaml` configuration
4. The `devfile.yaml` `deploy` command orchestrates the complete deployment process

## Running the application

### Local development
```bash
pip install -r requirements.txt
python app.py
```

### Using devfile
```bash
# Build and deploy
devfile deploy
```

## Additional resources
* For more information about Python, see [Python](https://www.python.org/)
* For more information about Flask, see [Flask](https://flask.palletsprojects.com/)
* For more information about devfiles, see [Devfile.io](https://devfile.io/)
* For more information about Dockerfiles, see [Dockerfile reference](https://docs.docker.com/engine/reference/builder/)