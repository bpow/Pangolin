FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY . .
RUN python3 -m pip install --upgrade -r requirements.txt \
  && python3 -m pip install . \
  && pip cache purge

ENTRYPOINT ["pangolin"]
