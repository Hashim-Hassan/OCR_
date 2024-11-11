# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
import cv2
import json, pathlib
import numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from doctr.file_utils import is_tf_available
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Enable GPU growth if using TF
if is_tf_available():
    import tensorflow as tf

    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    if any(gpu_devices):
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)


def main(args):
    model = ocr_predictor(args.detection, args.recognition, pretrained=True)
    
    image_file = None

    if args.path.lower().endswith(".pdf"):
        doc = DocumentFile.from_pdf(args.path)
    else:
        doc = DocumentFile.from_images(args.path)
        image_file = args.path
        
    out = model(doc)
    
    # json_out_file = open('Output/outjson.json', 'w')
    # texts = proto.Message.to_json(str(out))
    # print(type(out))
    # print(out)
    # exit()
    # json.dump(str(out), json_out_file, indent=4)
    # json_out_file.close()
    
    storeResult(args, image_file, out)
    
    for page in out.pages:
        page.show(block=not args.noblock, interactive=not args.static)
    
    print("Run successfully")
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="DocTR end-to-end analysis", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("path", type=str, default= "../Datasets/Client/YT_002_D.png", help="Path to the input document (PDF or image)")
    parser.add_argument("--detection", type=str, default="fast_base", help="Text detection model to use for analysis")
    parser.add_argument(
        "--recognition", type=str, default="crnn_vgg16_bn", help="Text recognition model to use for analysis"
    )
    parser.add_argument(
        "--noblock", dest="noblock", help="Disables blocking visualization. Used only for CI.", action="store_true"
    )
    parser.add_argument("--static", dest="static", help="Switches to static visualization", action="store_true")
    parser.add_argument("--jsonoutpath", type= str, default="Output/prediction.json", help="Predition json file out path")
    args = parser.parse_args()

    return args

def storeResult(args, image_file, result):
    
    image = cv2.imread(image_file)
    height, width = image.shape[:2]
    image = Image.open(image_file).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # print(result.pages[0].blocks[0].lines)
    
    # predictions = result.pages[0].predictions  
        
    os.makedirs(pathlib.Path(args.jsonoutpath).parent.resolve(), exist_ok=True)
    
    # output = defaultdict(dict)
    out_dict = {}
    # print(result)
    
    page_count = 0
    for page in result.pages:
        out_dict[f'Page-{page_count}'] = {}

        block_count = 0
        for block in page.blocks:
            out_dict[f'Page-{page_count}'][f'Block-{block_count}'] = {}

            line_count = 0
            for line in block.lines:
                out_dict[f'Page-{page_count}'][f'Block-{block_count}'][f'Line-{line_count}'] = []

                for word in line.words:
                    out_dict[f'Page-{page_count}'][f'Block-{block_count}'][f'Line-{line_count}'].append(word.value)
                    
                line_count += 1
            block_count += 1
        page_count += 1

    '''
    i=0
    for prediction in predictions["words"]:
        
        if prediction.confidence>0.05 and len(prediction.value)>1:
            temp = {}
            temp["label"] = prediction.value
            temp["coordinates"] = {}
            pixel_coordinates = (np.array([prediction.geometry]) * np.array([width, height])).astype(int)[0]

            temp["coordinates"]["x"] = int(pixel_coordinates[0][0])
            temp["coordinates"]["y"]  = int(pixel_coordinates[0][1])
            temp["coordinates"]["width"]  = int(pixel_coordinates[1][0]-pixel_coordinates[0][0])
            temp["coordinates"]["height"]  = int(pixel_coordinates[1][1]-pixel_coordinates[0][1])
            output[str(i)] = temp
            i+=1
            # print(temp)

            
            # cropped_image = image.crop((pixel_coordinates[0][0], pixel_coordinates[0][1], pixel_coordinates[1][0], pixel_coordinates[1][1]))
            # cropped_image.save(os.path.join(output_folder, f'Box-{i}.jpg'))
            draw.rectangle([tuple(pixel_coordinates[0]),tuple(pixel_coordinates[1])], outline="red", width=2)
            '''

    # image.save(os.path.join("Output", os.path.basename(image_file).split(".")[0]+".png"))

    with open(os.path.join("Output", os.path.basename(image_file).split(".")[0]+".json"), 'w') as fp:
        json.dump(out_dict, fp)

if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
