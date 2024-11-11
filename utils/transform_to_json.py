import os
import json

class ToJson:
    def __init__(self):
        pass
    
    def to_json(self, dataFile, outfile):
        annot = {
                    "image": "blank yellow tag.png",
                    "annotations": []
                }
        
        with open(dataFile) as file:
            for line in file:
                split = line.split(',')
                _coord = split[:8]
                _value = split[8:][-1].rstrip("\n")
                
                x = int(_coord[0])+int(_coord[4])/2
                y = int(_coord[1])+int(_coord[5])/2
                width = int(_coord[4])-int(_coord[0])
                height = int(_coord[5])-int(_coord[1])
                det = {
                        "label": f"{_value}",
                        "coordinates": {
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height
                        }
                    }
                annot["annotations"].append(det)
                
        with open(outfile, 'w') as file:
            json.dump(annot, file)
        