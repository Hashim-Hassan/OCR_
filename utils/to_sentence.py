import os
import json
from collections import defaultdict

class ToSentence:
    def __init__(self):
        self.detections = defaultdict(dict)

    def to_sentence(self, detection):
        word = detection['label']
        coordinates = detection['coordinates']
        x = coordinates['x']
        y = coordinates['y']
        width = coordinates['width']
        height = coordinates['height']

        xmin = x-width/2
        ymin = y-height/2
        xmax = x+width/2
        ymax = y+height/2
        
        max_key = 0
        flag = False
        for key, value in self.detections.items():
            max_key = key
            _word = value['label']
            _xmin = value['coordinates']['x'] - value['coordinates']['width']/2
            _ymin = value['coordinates']['y'] - value['coordinates']['height']/2
            _xmax = value['coordinates']['x'] + value['coordinates']['width']/2
            _ymax = value['coordinates']['y'] + value['coordinates']['height']/2

            if (value['coordinates']['y'] > y-width/10) and (value['coordinates']['y'] < y+width/10):
                flag = True

                if xmin < _xmin:
                    value['label'] = word + " " + _word
                else:
                    value['label'] = _word + " " + word
                
                xmin_ = min(xmin, _xmin)
                ymin_ = min(ymin, _ymin)
                xmax_ = max(xmax, _xmax)
                ymax_ = max(ymax, _ymax)

                x_ = (xmin_+xmax_)/2
                y_ = (ymin_+ymax_)/2
                width_ = xmax_ - xmin_
                height_ = ymax_ - ymin_

                value['coordinates']['x'] = int(x_)
                value['coordinates']['y'] = int(y_)
                value['coordinates']['width'] = int(width_)
                value['coordinates']['height'] = int(height_)
        
        if not flag:
            self.detections[max_key+1] = detection