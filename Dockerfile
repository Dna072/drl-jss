FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt && pip install pytest pillow

COPY . .

CMD ["python", "-c", "from stable_baselines3.common.env_checker import check_env; from custom_environment.environment_factory import init_custom_factory_env; check_env(init_custom_factory_env()); print('FactoryEnv OK')"]
