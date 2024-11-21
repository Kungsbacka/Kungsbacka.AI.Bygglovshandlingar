import pytesseract
import response_model as rm
import torch
import logging
import response_model as rm

class Direction:
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        self.valid_directions = ["SYDOST", "SYDVÄST", "SYDVAST", "NORDVÄST","NORDVAST","NORDOST","NORD", "NORR","SÖDER", "SODER", "SYD","ÖSTER", "OSTER","OST",  "ÖST","VÄSTER","VASTER", "VÄST", "VAST"]
        self.model = torch.load('models/riktning.model',  map_location=torch.device('cpu')).eval()
        self.calibration_cutof = 0.5
        

    def get_text(self, img):
        """Get text from image
        Args:
            img (PIL.Image): Image to get text from
        Returns:
            str: Text from image
        """

        text = pytesseract.image_to_osd(img)
        logging.info(f"Orientation: {text}")
        if "Orientation in degrees: 0" not in text:
            img = img.rotate(270, expand=True)
        text = pytesseract.image_to_string(img, lang='swe')
        return text

    def get_directions(self, img) -> list:
        """Get directions from image
        Args:
            img (PIL.Image): Image to get text from
        Returns:
            list: Directions found in image
        """
        text = self.get_text(img)
        found_directions = []
        text = text.upper()
        text = text.split("\n")
        for line in text:
            for direction in self.valid_directions:
                    
                    if direction in line:
                        found_directions.append(direction)
                        break

        logging.info(f"Directions identified: {found_directions}")
        return set(found_directions)
    
    #Validates directions in the image
    def validate_directions_text_fasad(self, img, fasader) -> rm.ComponentResponse:
        """Validate directions in image
        Args:
            img (PIL.Image): Image to validate
            fasader (list): List of fasader
        Returns:
            rm.ComponentResponse: Response with status, code and message
        """
        
        found_directions = self.get_directions(img)
        logging.info(f'Found directions: {found_directions}')
        if len(found_directions) == 4:
            logging.info('Success: 4 directions found')
            return rm.ComponentResponse(status=True, code="200", msg="Success")
        elif len(found_directions) == len(fasader):
            logging.info('Success: Number of directions matches number of fasader')
            return rm.ComponentResponse(status=True, code="200", msg="Success")
        else:
            if len(found_directions) > 4:
                logging.error('Error: More than 4 directions found')
                return rm.ComponentResponse(status=False, code="220", msg="More than 4 directions found")
            elif len(found_directions) < len(fasader):
                logging.error('Error: More fasader than directions found')
                return rm.ComponentResponse(status=False, code="240", msg="More fasader than directions found")
            elif len(found_directions) > len(fasader):
                logging.error('Error: More directions than fasader found')
                return rm.ComponentResponse(status=False, code="250", msg="More directions than fasader found")

    #Find direction symbol
    def find_direction_Symbol(self, img) -> rm.ComponentResponse:
        """Find direction symbol in image
        Args:
            img (np.array): Image to find direction symbol in
        Returns:
            rm.ComponentResponse: Response with status, code and message
        """
        img = img / 255.0
        print(img.shape)
        img = img.transpose((2, 0, 1))
        with torch.no_grad():

            predicton = self.model([torch.from_numpy(img)])
            if len(predicton[0]["scores"]) == 0:
                logging.error("No predictions")
                return {"status":False, "code":"210", "msg":"No direction symbol found"}
            if predicton[0]['scores'][0] > self.calibration_cutof:
                return {"status":True, "code":"200", "msg":"Success"}
            else:
                return {"status":False, "code":"210", "msg":"No direction symbol found"}
        
    