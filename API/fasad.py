import torch 
import torchvision
import logging

class Fasad_model:
    def __init__(self):
        print("INIT FASAD MODEL")
        self.model = torch.load('models/fasad.model',  map_location=torch.device('cpu')).eval()
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
    
    def get_fasader(self, data):
        """
        Inputs
            data: np.array
        Returns
            fasader: list
        """
        data = data / 255.0
        logging.info(data.shape)
        data = data.transpose((2, 0, 1))
        with torch.no_grad():

            raw_prediction = self.model([torch.from_numpy(data)])
            logging.info("Raw predictions: " + str(raw_prediction))
            prediction = self.decode_prediction(raw_prediction[0], self.calibration_cutof)
            logging.info("Final predictions:" + str(prediction["boxes"]))
            logging.info(str(prediction["boxes"]))
            logging.info(str(prediction["scores"]))
            logging.info("Number of fasader: " + str(len(prediction["boxes"])) + " found")
            return prediction["boxes"]


