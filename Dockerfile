FROM docker.uclv.cu/python:3.11.4-slim

WORKDIR /roop

COPY . .

RUN pip install -r requirements.txt

RUN python run.py --target ./.github/examples/model.png  --source ./.github/examples/face.jpeg -o ./.github/examples/swapped.png --execution-provider cuda --frame-processor face_swapper face_enhancer

ENTRYPOINT ["python", "run.py"]