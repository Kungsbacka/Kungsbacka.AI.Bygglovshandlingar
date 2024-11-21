import torch
import numpy as np
import response_model as rm
import logging

class SkalStock:
    def __init__(self):
        self.model = torch.load('models/skalstock.model',  map_location=torch.device('cpu')).eval()
        self.calibration_cutof = 0.25

    def predict(self, data) -> rm.ComponentResponse:
        """
        Inputs
            data: np.array
        Returns
            response: rm.ComponentResponse
        """

        data = data / 255.0
        print(data.shape)
        data = data.transpose((2, 0, 1))
        with torch.no_grad():

            predicton = self.model([torch.from_numpy(data)])
            if len(predicton[0]["scores"]) == 0:
                logging.error("No predictions")
                return rm.ComponentResponse(status=False, code="110", msg="No scale was found")
            logging.info(str(predicton[0]['scores'][0]))
            if predicton[0]['scores'][0] > self.calibration_cutof:
                logging.info("Skalstock ok")
                return rm.ComponentResponse(status=True, code="100", msg="Success")
            else:
                logging.error(f"Skalstock error: {predicton[0]}")
                return rm.ComponentResponse(status=False, code="110", msg="No scale was found above given sensitivity")