import logging.handlers
import fastapi
from fastapi import UploadFile, File, FastAPI, HTTPException
import response_model as rm
import uvicorn
import sys
from pdf2image import convert_from_path, convert_from_bytes
from skalstock import SkalStock
from direction import Direction
from fasad import Fasad_model
from ground_level import GroundLevel
from distance_markers import DistanceMarkers
from situationsplan import validate_situationsplan
import numpy as np
import traceback
from contextlib import asynccontextmanager
import logging

from logging.handlers import TimedRotatingFileHandler
from lib import save_file, check_if_folder_exists, setup_logging
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


models = {}
logger = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up")
    logger = setup_logging()

    logging.info("Starting up")
    if not check_if_folder_exists("/data"):
        logging.exception("Data folder does not exist or is not mounted correctly") 
        raise("Data folder does not exist or is not mounted correctly")
    logging.info("Loading models")
    skalstock = SkalStock()
    direction = Direction()
    fasad = Fasad_model()
    ground = GroundLevel()
    distance_markers = DistanceMarkers()
    models["skalstock"] = skalstock
    models["direction"] = direction
    models["fasad"] = fasad
    models["ground"] = ground
    models["distance_markers"] = distance_markers

    logging.info("Models loaded")
    yield
    await on_shutdown()

def on_shutdown():
    pass

app = fastapi.FastAPI(lifespan=lifespan)

@app.post("/checkFasadritning")
async def check_fasad(file: UploadFile = File(...), id:str = None) -> rm.FasadritningResponse:
    # Process the uploaded PDF file here
    # You can access the file using the `file` parameter
    
    # Example: Read the contents of the file
    logging.info("Checking fasad requested")
    if id == None:
        id = ""
    try:
        contents = await file.read()
        pdf = convert_from_bytes(contents)
        
        if len(pdf) > 1:
            logging.error("More than one page in file")
            raise HTTPException(status_code=422, detail="More than one page")
        save_file(np.array(pdf[0], dtype=np.float32))

        skalstock_res = models["skalstock"].predict(np.array(pdf[0], dtype=np.float32))
        fasader = models["fasad"].get_fasader(np.array(pdf[0], dtype=np.float32))
        direction_marker_response = models["direction"].validate_directions_text_fasad(pdf[0], fasader)
        ground_leval_ok = models["ground"].validate_ground_lines(pdf[0], fasader)

        logging.info(f"Skalstock: {skalstock_res}")
        logging.info(f"Ground level: {ground_leval_ok}")
        logging.info(f"Direction marker: {direction_marker_response}")

        return {
                    "scale":skalstock_res,
                    "ground_level":ground_leval_ok,
                    "direction_marker":direction_marker_response,
                }
        
    except Exception as e:
        logging.error(f"Error for ID:{id} in chechFasad: {e}")
        logging.exception(f"Error for ID:{id} in checkFasadritning")
        return {
                    "scale":{"status":False,"code":"422","msg":"Error"},
                    "ground_level":{"status":False,"code":"422", "msg":"Error"},
                    "direction_marker":{"status":False,"code":"422", "msg":"Error"},
                }
    
    
    


@app.post("/checkSituationsplan")
async def check_situation(file: UploadFile = File(...), id:str = None) -> rm.SituationsplanResponse:
    # Process the uploaded PDF file here
    # You can access the file using the `file` parameter
    
    # Example: Read the contents of the file
    if id == None:
        id = ""
    try:
        logging.info("Checking fasad requested")
        contents = await file.read()
        pdf = convert_from_bytes(contents)
        print(pdf)
        if len(pdf) > 1:
            logging.error("More than one page in file")
            raise HTTPException(status_code=422, detail="More than one page")

        save_file(np.array(pdf[0], dtype=np.float32))

        



        skalstock_res = models["skalstock"].predict(np.array(pdf[0], dtype=np.float32))
        distance_messurement = models["distance_markers"].find_distance_markers(np.array(pdf[0], dtype=np.float32))
        direction_marker_response = models["direction"].find_direction_Symbol(np.array(pdf[0], dtype=np.float32))

        logging.info(f"Skalstock: {skalstock_res}")
        logging.info(f"Distance messurement: {distance_messurement}")
        logging.info(f"Direction marker: {direction_marker_response}")

        return {
                    "scale":skalstock_res,
                    "distance_messurement":distance_messurement,
                    "direction_marker":direction_marker_response,
                }
        
        


    except Exception as e:
        logging.error(f"Error for ID:{id} in checkSituationsplan: {e}")
        logging.exception(f"Error for ID:{id} in checkSituationsplan")
        return {
                    "scale":{"status":False,"code":"422","msg":"Error"},
                    "distance_messurement":{"status":False,"code":"422", "msg":"Error"},
                    "direction_marker":{"status":False,"code":"422", "msg":"Error"},
                }
    

@app.post("/checkPlanritning")
async def check_plan(file: UploadFile = File(...), id:str = None) -> rm.PlanritningResponse:
    # Process the uploaded PDF file here
    # You can access the file using the `file` parameter
    
    # Example: Read the contents of the file
    if id == None:
        id = ""
    try:
        logging.info("Checking fasad requested")
        contents = await file.read()
        pdf = convert_from_bytes(contents)
        print(pdf)
        if len(pdf) > 1:
            logging.error("More than one page in file")
            raise HTTPException(status_code=422, detail="More than one page")
        save_file(np.array(pdf[0], dtype=np.float32))
        skalstock_res = models["skalstock"].predict(np.array(pdf[0], dtype=np.float32))
        
        logging.info(f"Skalstock: {skalstock_res}")
        return {
                    "scale":skalstock_res,
                }
    except Exception as e:
        logging.error(f"Error for ID:{id} in checkPlanritning: {e}")
        logging.exception(f"Error for ID:{id} in checkPlanritning")
        return {
                    "scale":{"status":False,"code":"422","msg":e},
                }


@app.post("/checkSektionsritning")
async def check_sektion(file: UploadFile = File(...), id:str = None) -> rm.SektionsritningResponse:
    # Process the uploaded PDF file here
    # You can access the file using the `file` parameter
    
    # Example: Read the contents of the file
    if id == None:
        id = ""
    try:
        logging.info("Checking fasad requested")
        contents = await file.read()
        pdf = convert_from_bytes(contents)
        print(pdf)
        if len(pdf) > 1:
            raise HTTPException(status_code=422, detail="More than one page")
        save_file(np.array(pdf[0], dtype=np.uint8))
        skalstock_ok = models["skalstock"].predict(np.array(pdf[0], dtype=np.float32))
        if skalstock_ok:
            return {
                "scale":{"status":True,"code":"100","msg":"Success"},
            }
        else:
            return {
                "scale":{"status":False,"code":"110","msg":"No scale was found"},
            }

    except Exception as e:
        logging.error(f"Error for ID:{id} in checkSektionsritning: {e}")
        logging.exception(f"Error for ID:{id} in checkSektionsplan")
        return {
                    "scale":{"status":False,"code":"422","msg":e},
                }
    
    

if __name__ == "__main__":

    uvicorn.run("api:app", port=int(
        sys.argv[1]), host=sys.argv[2],log_level="debug", workers=1)
