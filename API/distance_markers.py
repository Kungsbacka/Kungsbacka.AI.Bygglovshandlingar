import torch
import response_model as rm 
import logging
import torchvision
from situationsplan import validate_situationsplan

class DistanceMarkers:
    def __init__(self, ):
        print("INIT Avståndsmodell")
        self.model = torch.load('models/distance_marker.model',  map_location=torch.device('cpu')).eval()
        self.calibration_cutof = 0.6

    def decode_prediction(self, prediction, 
                        score_threshold = 0.8, 
                        nms_iou_threshold = 0.2):
        """
        Inputs
            prediction: dict
            score_threshold: float
            nms_iou_threshold: float
        Returns
            prediction: tuple
        """
        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["labels"]
        # Remove any low-score predictions.
        if score_threshold is not None:
            want = scores > score_threshold
            boxes = boxes[want]
            scores = scores[want]
            labels = labels[want]
        # Remove any overlapping bounding boxes using NMS.
        if nms_iou_threshold is not None:
            want = torchvision.ops.nms(boxes = boxes, scores = scores, 
                                    iou_threshold = nms_iou_threshold)
            boxes = boxes[want]
            scores = scores[want]
            labels = labels[want]
        return {"boxes": boxes, 
                "labels": labels, 
                "scores": scores}

    def find_distance_markers(self, img) -> rm.ComponentResponse:
        #correct_situationsplan = validate_situationsplan(img)
        logging.info("Finding distance markers")
        img = img / 255.0
        print(img.shape)
        img = img.transpose((2, 0, 1))
        with torch.no_grad():

            raw_prediction = self.model([torch.from_numpy(img)])
            logging.info("Raw predictions: " + str(raw_prediction))
            prediction = self.decode_prediction(raw_prediction[0], self.calibration_cutof)
            logging.info("Final predictions:" + str(prediction["boxes"]))
            #if not correct_situationsplan:
            #    logging.error("Situationsplan not correct")
            #   return rm.ComponentResponse(status=False, code="410", msg="Situationsplan not correct")
            logging.info("Number of distance markers: " + str(len(prediction["boxes"])) + " found")
            num = len(prediction["boxes"])
            if num == 0:
                logging.error("No Distance lines found")
                return rm.ComponentResponse(status=False, code="440", msg="No Distance symbol found")
            if num == 1:
                logging.info("One Distance symbol found")
                return rm.ComponentResponse(status=False, code="460", msg="One Distance symbol found")
            if num == 2:
                logging.info("Two Distance symbol found")
                return rm.ComponentResponse(status=False, code="470", msg="Two Distance symbol found")
            if num >= 3:
                
                logging.info(f"Distance symbol found: {num} was found")
                return rm.ComponentResponse(status=True, code="400", msg="Success")
            

            
            