"""
Run:  uvicorn main:app --port 8000
Needs: main.py, rf_model.py, random_forest_model.pkl in the same folder.
"""
import os
import pickle
from contextlib import asynccontextmanager
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

import rf_model

MODEL_PATH = os.getenv("MODEL_PATH", "random_forest_model.pkl")
LABEL_COLS = ["behavior", "composition"]

# Placeholder datapoint (labels removed; empty cells -> null)
HEADER = "v_rms_start,i_rms_start,p_start,q_start,pf_start,z_start,v_rms_end,i_rms_end,p_end,q_end,pf_end,z_end,s_end,ratio_v,ratio_i,ratio_p,phi_end,z_r_eq,z_x_eq,thd_v,thd_i,dv_span,n_i,n_i_ep,n_s,n_p,n_q,n_phi,n_z_rms,n_g,n_b,n_i_resid,ih2,ih3,ih4,ih5,ih6,ih7,ih8,ih9,g_3,b_3,yr_3,dphi_3,vh_3,g_5,b_5,yr_5,dphi_5,vh_5,g_7,b_7,yr_7,dphi_7,vh_7,g_1,b_1,yslope,phase_slope,liss_area,liss_slope"
VALUES = "234.66607350511575,55.690110315103176,13058.108005739052,-41.81138041519198,0.9991987258480265,4.213783599589495,208.52306469181838,55.75569959319816,11617.052777096096,-22.764766442048455,0.9992003873415871,3.7399416779491568,11626.365499717218,0.8885948513016376,1.0011777545011649,0.8896428772062834,-0.1123949705894489,3.739934482081192,-0.007336497886553999,0.0357819698473803,0.03493402434383978,-0.11864369565233698,0.03290848318325579,0.02671853330108625,1.0329084831832576,1.032902363249007,,0.35909196227140644,0.9663790491621791,-0.9670976367471594,,0.003218415485915815,0.0025978738419214636,0.017380233286884973,0.004044885493362638,0.0255154532495287,0.0023224044846184427,0.005844763451913216,0.004030259203977024,0.003948102482803138,0.9424112267390906,-0.06078331472945478,-0.10990257776644156,-3.755980756161171,0.01908878516405312,0.9323642071778553,-0.01891648940823362,-0.12321458605257138,-0.9885848647607215,0.02731413339921568,-0.2912852409119565,-1.0762670373148742,0.5037021281681574,-58.091721710475014,0.0035917318672338333,0.999990703263301,0.0002880084656022067,0.159656326267356,-21.046546769624374,-0.0040249901320191205,0.9991982194739419"
EXAMPLE = {
    c: (float(v) if v else None)
    for c, v in zip(HEADER.split(","), VALUES.split(","), strict=True)
}


class _Unpickler(pickle.Unpickler):
    """The pickle references classes in the training script's __main__."""

    def find_class(self, module, name):
        if module == "__main__" and hasattr(rf_model, name):
            return getattr(rf_model, name)
        return super().find_class(module, name)


art: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(MODEL_PATH, "rb") as f:
        art.update(_Unpickler(f).load())
    yield


app = FastAPI(title="Random Forest API", lifespan=lifespan)


class PredictRequest(BaseModel):
    features: Dict[str, Optional[float]] = Field(
        ..., json_schema_extra={"example": EXAMPLE}
    )


@app.post("/predict")
def predict(req: PredictRequest):
    # Same column order and NaN handling as training
    x = np.array(
        [[req.features.get(c) for c in art["feature_columns"]]], dtype=np.float64
    )  # None -> NaN
    x = np.nan_to_num(x, nan=art["nan_fill"], posinf=1e12, neginf=-1e12)

    out = {}
    for label in LABEL_COLS:
        idx = int(art["models"][label].predict(x)[0])
        out[label] = art["label_mappings"][label]["classes"][idx]
    out["joint"] = f"{out['behavior']}-{out['composition']}"
    return out