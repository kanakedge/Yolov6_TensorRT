from ast import In
import os
import json
import glob
from aws_utils import aws_util


ONNX_EXPORT = "./deploy/ONNX/export_onnx.py"
input_config = json.loads(open("./opt_params.json",'r').read())
artifacts_uri = input_config["artifacts_path"]

modelname = input_config["modelname"]
sessionid = input_config["sessionid"]
BATCH_SIZE = 16

ARTIFACTS_LOCATION = "./artifacts.zip"

MODEL_NAME = modelname
MODEL_PATH=f"/tmp/{MODEL_NAME}"
OPT_MODEL_PATH = MODEL_PATH + "_out"
CALIB_DATA = ""
CALIB_FILE = ""
fp16 = False
int8 = False
# os.system(f"curl -o {ARTIFACTS_LOCATION} \"{artifacts_uri}\"")

assert os.path.isfile(ARTIFACTS_LOCATION) , "artifacts not downloaded"

os.system(f"unzip -oqq {ARTIFACTS_LOCATION} -d {MODEL_PATH}")
os.system(f"mkdir {OPT_MODEL_PATH}")

model_config = json.loads(open(f"{MODEL_PATH}/train_params.json").read())
try:
    model_config = model_config['config_json']
    if isinstance(model_config, str):
        model_config = json.loads(model_config)
except Exception as _:
    raise
INPUT_SIZE = model_config["img_size"]

assert isinstance(INPUT_SIZE, list), f"{INPUT_SIZE} not of type list"
PYTORCH_WEIGHT = os.path.join(MODEL_PATH, "final_model.pt")
ONNX_WEIGHT = os.path.join(MODEL_PATH, f"final_model.onnx")

# print(INPUT_SIZE, type(INPUT_SIZE), type(INPUT_SIZE[0]))
# INPUT_SIZE = "[640, 640]"

os.system(f"python {ONNX_EXPORT} --weights {PYTORCH_WEIGHT} --img-size {INPUT_SIZE[0]} --batch-size {BATCH_SIZE}")
assert os.path.isfile(ONNX_WEIGHT), "Error during conversion, onnx file not found!"
conversion = f"/usr/src/tensorrt/bin/trtexec --onnx={ONNX_WEIGHT} --verbose --avgRuns=1 --exportTimes=performance.json"

if fp16:
    output_engine = os.path.join(OPT_MODEL_PATH, f"final_fp16_bs{BATCH_SIZE}.engine")
    conversion += f" --fp16 --saveEngine={output_engine}"

elif int8:
    output_engine = os.path.join(OPT_MODEL_PATH, f"final_int8_bs{BATCH_SIZE}.engine")
    conversion += f" --int8 --saveEngine={output_engine} --calib={CALIB_FILE}"

else:
    output_engine = os.path.join(OPT_MODEL_PATH, f"final_fp32_bs{BATCH_SIZE}.engine")
    conversion += f" --saveEngine={output_engine}"

os.system(conversion)







# print("ONNX File converted and found")

# aws_U = aws_util()
# aws_U.upload_file(f"{OPT_MODEL_PATH}/optimize_stats.csv","enap-optimize-data",sessionid+"/optimize_stats.csv")
# os.system(f"zip -r /tmp/artifacts.zip {OPT_MODEL_PATH}/")
# aws_U.upload_file("/tmp/artifacts.zip","enap-optimize-data",sessionid+"/artifacts.zip")




