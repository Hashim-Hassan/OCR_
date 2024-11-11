## Steps to run:

### 1. Pull docker image:
        docker pull ghcr.io/mindee/doctr:torch-py3.9.18-gpu-2024-10

### 2. Run Docker:
        docker run -it --shm-size=6gb --network host --gpus all --user 0 -v /temp/.X11-unix:/temp/.X11-unix -e DISPLAY=$DISPLAY -v <path to ocr dir>:/home/ocr ghcr.io/mindee/doctr:torch-py3.9.18-gpu-2024-10 bash
   
### 3. Install dependencies:
        cd home/ocr
        pip install -r requirements.txt

### 4. Run:
        Assuming current dir as: /home/ocr
        python3 tests/test_ocr.py

### 5. Output:
        OCR output will be stored at dir: ocr/output
        subdirectories: 
            images - contains images with drawn bounding boxes over recognized characters
            json - contains json file detections,
            accuracy - accuracy metrics for all doctr models
                Note: due to some special characters and undefident speces and line breaks accuracy may drop as groundtruth is not uptomar,
                if GT is manually corrected, accuracy will increase significantly.  

