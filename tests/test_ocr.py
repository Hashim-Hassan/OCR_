import json
# from PIL import Image,ImageDraw
import cv2
import numpy as np

import os
import sys
from pathlib import Path
# os.chdir('../')
sys.path[0] = str(Path(__file__).parent.parent)
from doctrv.doctr.io import DocumentFile
from doctrv.doctr.models import kie_predictor
from utils.transform_to_json import ToJson
from utils.to_sentence import ToSentence
from utils.accuracy import Accuracy

class TestOcr:
    def __init__(self):
        pass

    def get_ocr_detections(self, model, imageFile):
        image = cv2.imread(imageFile)
        height, width = image.shape[:2]

        doc = DocumentFile.from_images(imageFile)
        result = model(doc)
        predictions = result.pages[0].predictions

        toSentenceObj = ToSentence()

        for prediction in predictions["words"]:
            if prediction.confidence>0.05 and len(prediction.value)>1:
                temp = {}
                temp["label"] = prediction.value
                temp["coordinates"] = {}
                pixel_coordinates = (np.array([prediction.geometry]) * np.array([width, height])).astype(int)[0]

                temp["coordinates"]["x"] = int((pixel_coordinates[0][0]+pixel_coordinates[1][0])/2)
                temp["coordinates"]["y"]  = int((pixel_coordinates[0][1]+pixel_coordinates[1][1])/2)
                temp["coordinates"]["width"]  = int((pixel_coordinates[1][0]-pixel_coordinates[0][0]))
                temp["coordinates"]["height"]  = int(pixel_coordinates[1][1]-pixel_coordinates[0][1])
                
                toSentenceObj.to_sentence(temp)
        return toSentenceObj.detections
    
    def draw_detections(self, imageFile, detections):
        image = cv2.imread(imageFile)
        for value in detections.values():
            x_min = int(value['coordinates']['x'] - value['coordinates']['width']/2)
            y_min = int(value['coordinates']['y'] - value['coordinates']['height']/2)
            x_max = int(value['coordinates']['x'] + value['coordinates']['width']/2)
            y_max = int(value['coordinates']['y'] + value['coordinates']['height']/2)

            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        return image
    
    def test_accuracy(self, gt_dir, detoutDir, reportOutDir):
        accuracyObj = Accuracy()
        
        for img_dir in os.listdir(detoutDir):
            for file in os.listdir(f'{detoutDir}/{img_dir}'):
                if file.endswith('.json'):
                    ocr_json_path = f'{detoutDir}/{img_dir}/{file}'
                    gt_json_path = f"{gt_dir}/{img_dir}.json"
                    
                    det_model, recog_model = file.strip('.json').split('-')

                    # Load the OCR and Ground Truth JSON files
                    ocr_data = accuracyObj.parseOCRoutput(ocr_json_path)
                    gt_data = accuracyObj.loadJson(gt_json_path)

                    # Evaluate the text accuracy
                    avgCER, avgWER, exactMatchRate, exactMatchedTexts = accuracyObj.evaluateAccuracy(ocr_data, gt_data)
                    accuracy_values = avgCER, avgWER, exactMatchRate, exactMatchedTexts
                    accuracyObj.accuracyDictionary(det_model, recog_model, accuracy_values)

            # accuracyObj.accuracyMatrix()
            excelOutFile = f'{reportOutDir}/{img_dir}.xlsx'
            accuracyObj.save_to_excel(excelOutFile)

def main():
    
    input_img_dir = "Datasets/img"
    csv_dir = "Datasets/box"
    groundTruthDir = "Datasets/groundTruth"

    detDrawnOutDir = "output/images"
    detOutDir = "output/json"
    det_models = ['linknet_resnet18', 'linknet_resnet34', 'linknet_resnet50', 'db_resnet50', 'db_mobilenet_v3_large', 'fast_tiny', 'fast_small', 'fast_base']
    recog_models = ['crnn_vgg16_bn', 'crnn_mobilenet_v3_small', 'crnn_mobilenet_v3_large', 'sar_resnet31', 'master', 'vitstr_small', 'vitstr_base', 'parseq']

    testOcrObj = TestOcr()

    for file in os.listdir(input_img_dir):
        if file.split('.')[-1] != 'jpg':
            continue
        gt_csv_file = f'{csv_dir}/{file.split(".")[0]}.csv'
        image_file = f'{input_img_dir}/{file}'
        
        json_out_file = f'{groundTruthDir}/{file.split(".")[0]}.json'
        os.makedirs(groundTruthDir, exist_ok=True)
        ToJson().to_json(gt_csv_file, json_out_file)

        for det_model in det_models:
            for recog_model in recog_models:
                model = kie_predictor(det_arch= det_model, reco_arch= recog_model, pretrained=True)
                detections = testOcrObj.get_ocr_detections(model, image_file)
                detDrawnImg = testOcrObj.draw_detections(image_file, detections)

                imgOutPath = os.path.join(detDrawnOutDir,os.path.basename(image_file).split(".")[0])
                os.makedirs(imgOutPath, exist_ok=True)
                cv2.imwrite(os.path.join(imgOutPath, f"{det_model}-{recog_model}.png"), detDrawnImg)

                json_out_path = os.path.join(detOutDir, os.path.basename(image_file).split(".")[0])
                os.makedirs(json_out_path, exist_ok=True)
                with open(os.path.join(json_out_path, f"{det_model}-{recog_model}.json"), 'w') as fp:
                    json.dump(detections, fp)
    
    gt_dir = "Datasets/groundTruth"
    reportOutDir = "output/accuracy"
    testOcrObj.test_accuracy(gt_dir, detOutDir, reportOutDir)

        
if __name__ == '__main__':
    main()

