wget --no-check-certificate https://modelrelease.blob.core.windows.net/mass/zhen_mass_pre-training.pt -P ./MASS/model

gunicorn --config ./conf/gunicorn_config.py src:app
