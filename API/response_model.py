from pydantic import BaseModel

class ComponentResponse(BaseModel):
    status:bool
    code:str
    msg:str
    class Config:
        json_schema_extra = {
            "example": {
                "status": True,
                "code": "100",
                "msg": "Successfull"
            }
        }

class FasadritningResponse(BaseModel):
    scale: ComponentResponse
    ground_level: ComponentResponse
    direction_marker: ComponentResponse
    class Config:
        json_schema_extra = {
            "example":{
                "scale":ComponentResponse.model_json_schema(),
                "ground_level":ComponentResponse.model_json_schema(),
                "direction_marker":ComponentResponse.model_json_schema()
            }
        }

class SituationsplanResponse(BaseModel):
    scale: ComponentResponse
    distance_messurement: ComponentResponse
    direction_marker: ComponentResponse
    class Config:
        json_schema_extra = {
            "example":{
                "scale":ComponentResponse.model_json_schema(),
                "distance_messurement":ComponentResponse.model_json_schema(),
                "direction_marker":ComponentResponse.model_json_schema()
            }
        }

class PlanritningResponse(BaseModel):
    scale: ComponentResponse
    class Config:
        json_schema_extra = {
            "example":{
                "scale":ComponentResponse.model_json_schema(),
            }
        }

class SektionsritningResponse(BaseModel):
    scale: ComponentResponse
    class Config:
        json_schema_extra = {
            "example":{
                "scale":ComponentResponse.model_json_schema(),
            }
        }

