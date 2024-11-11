import os, json
import re
import Levenshtein
import pandas as pd
import numpy as np

from scipy.spatial import distance

class Accuracy:
    def __init__(self):
        self.distanceThreshold = 10
        
        self.avgCER_dict = {}
        self.avgWER_dict = {}
        self.exactMatchRate_dict = {}
        
    def parseOCRoutput(self, json_file_path):
        ocr_data = []
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            for key in data:
                ocr_data.append(data[key])
        return ocr_data

    def loadJson(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        return data['annotations']

    def CER(self, ocr_text, gt_text):
        return Levenshtein.distance(ocr_text, gt_text) / max(len(ocr_text), len(gt_text))

    def WER(self, ocr_text, gt_text):
        return Levenshtein.distance(ocr_text.split(), gt_text.split()) / len(gt_text.split())

    def exactMatch(self, ocr_text, gt_text):
        return 1 if ocr_text.strip() == gt_text.strip() else 0

    def calculateCenter(self, coordinates):
        x_center = coordinates['x'] + coordinates['width'] / 2
        y_center = coordinates['y'] + coordinates['height'] / 2
        return (x_center, y_center)

    # Find the closest annotation based on the center coordinates
    def findClosestAnnotation(self, ocr_annotation, gt_annotations):
        ocr_center = self.calculateCenter(ocr_annotation['coordinates'])
        closestAnnotation = None
        min_distance = float('inf')

        for gt_annotation in gt_annotations:
            gt_center = (gt_annotation['coordinates']['x'], gt_annotation['coordinates']['y'])
            dist = distance.euclidean(ocr_center, gt_center)
            if dist < min_distance and dist < self.distanceThreshold:
                min_distance = dist
                closestAnnotation = gt_annotation

        return closestAnnotation

    # Calculate accuracy metrics between OCR and ground truth
    def evaluateAccuracy(self, ocr_data, gt_data):
        totalCER = 0
        totalWER = 0
        exactMatches = 0
        total = 0
        exactMatchedTexts = []

        for ocr_item in ocr_data:
            ocr_text = ocr_item['label']

            closest_gt_annotation = self.findClosestAnnotation(ocr_item, gt_data)
            if closest_gt_annotation:
                gt_text = closest_gt_annotation.get('label')

                # Calculate Character Error Rate (CER)
                charAcc = self.CER(ocr_text, gt_text)
                totalCER += charAcc

                # Calculate Word Error Rate (WER)
                wordAcc = self.WER(ocr_text, gt_text)
                totalWER += wordAcc

                # Check for Exact Match
                flag = self.exactMatch(ocr_text, gt_text)
                if flag == 1:
                    exactMatches += 1
                    exactMatchedTexts.append(ocr_text.strip())
                else:
                    pass

                total += 1

        # Calculate average CER, WER, and Exact Match Rate
        avgCER = totalCER / total if total > 0 else 0
        avgWER = totalWER / total if total > 0 else 0
        exactMatchRate = exactMatches / total if total > 0 else 0

        return avgCER, avgWER, exactMatchRate, exactMatchedTexts

    def findClosestAnnotation2(self, gt_annotation, ocr_annotations):
        gt_center = (gt_annotation['coordinates']['x'], gt_annotation['coordinates']['y'])
        closestAnnotation = None
        min_distance = float('inf')

        for ocr_annotation in ocr_annotations:
            ocr_center = self.calculateCenter(ocr_annotation['coordinates'])
            dist = distance.euclidean(gt_center, ocr_center)
            if dist < min_distance and dist < self.distanceThreshold:
                min_distance = dist
                closestAnnotation = ocr_annotation

        return closestAnnotation

    # using each GT box with all the ocr output for comparison
    def evaluateAccuracy2(self, ocr_data, gt_data):
        totalCER = 0
        totalWER = 0
        exactMatches = 0
        total = 0
        exactMatchedTexts = []

        for gt_item in gt_data:
            gt_text = gt_item['label']

            closest_gt_annotation = self.findClosestAnnotation2(gt_item, ocr_data)
            if closest_gt_annotation is not None:
                ocr_text = closest_gt_annotation.get('label')

                # Calculate Character Error Rate (CER)
                charAcc = self.CER(ocr_text, gt_text)
                totalCER += charAcc

                # Calculate Word Error Rate (WER)
                wordAcc = self.WER(ocr_text, gt_text)
                totalWER += wordAcc

                # Check for Exact Match
                flag = self.exactMatch(ocr_text, gt_text)
                if flag == 1:
                    exactMatches += 1
                    exactMatchedTexts.append(ocr_text.strip())
                else:
                    pass

                total += 1

        # Calculate average CER, WER, and Exact Match Rate
        avgCER = totalCER / total if total > 0 else 0
        avgWER = totalWER / total if total > 0 else 0
        exactMatchRate = exactMatches / total if total > 0 else 0
        
        return avgCER, avgWER, exactMatchRate, exactMatchedTexts
    
    def accuracyDictionary(self, det_model, recog_model, accuracy_values):             
        if det_model in self.avgCER_dict:
            self.avgCER_dict[det_model][recog_model] = accuracy_values[0]
        else:
            self.avgCER_dict[det_model] = {recog_model:accuracy_values[0]}
            
        if det_model in self.avgWER_dict:
            self.avgWER_dict[det_model][recog_model] = accuracy_values[1]
        else:
            self.avgWER_dict[det_model] = {recog_model:accuracy_values[1]}
            
        if det_model in self.exactMatchRate_dict:
            self.exactMatchRate_dict[det_model][recog_model] = accuracy_values[2]
        else:
            self.exactMatchRate_dict[det_model] = {recog_model:accuracy_values[2]}
    
    def accuracyMatrix(self):
        det_index = 0
        for det in self.accuracyDict:
            self.det_models.append(det)
            self.avgCER_Matrix.append([])
            self.avgWER_Matrix.append([])
            self.exactMatchRate_Matrix.append([])
            for recog in self.accuracyDict[det]:
                if recog not in self.recog_models:
                    self.recog_models.append(recog)
                self.avgCER_Matrix[det_index].append(self.accuracyDict[det][recog][0])
                self.avgWER_Matrix[det_index].append(self.accuracyDict[det][recog][1])
                self.exactMatchRate_Matrix[det_index].append(self.accuracyDict[det][recog][2])  
            det_index += 1
            
    def accuracyMatrix1(self, det_index, accuracyValues):
        self.avgCER_Matrix[det_index].append(accuracyValues[0])
        self.avgWER_Matrix[det_index].append(accuracyValues[1])
        self.exactMatchRate_Matrix[det_index].append(accuracyValues[2])  
    
    def save_to_excel(self, excelOutFile):
        writter = pd.ExcelWriter(excelOutFile)
        avgCER_df = pd.DataFrame(self.avgCER_dict)
        avgWER_df = pd.DataFrame(self.avgWER_dict)
        exactMatchRate_df = pd.DataFrame(self.exactMatchRate_dict)
        avgCER_df.to_excel(writter, sheet_name='avgCER')
        avgWER_df.to_excel(writter, sheet_name='avgWER')
        exactMatchRate_df.to_excel(writter, sheet_name='exactMatchRate')
        writter.close()
        self.clearVar()
        
    
    def clearVar(self):
        self.avgCER_dict = {}
        self.avgWER_dict = {}
        self.exactMatchRate_dict = {}
            