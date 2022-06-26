from pydantic import BaseModel, Field
import os
from pathlib import Path
from enum import Enum
from typing import Any
from synthesizer.hparams import hparams
from synthesizer.train import train as synt_train

# Constants
SYN_MODELS_DIRT = f"synthesizer{os.sep}saved_models"
ENC_MODELS_DIRT = f"encoder{os.sep}saved_models"


# EXT_MODELS_DIRT = f"ppg_extractor{os.sep}saved_models"
# CONV_MODELS_DIRT = f"ppg2mel{os.sep}saved_models"
# ENC_MODELS_DIRT = f"encoder{os.sep}saved_models"

# Pre-Load models
if os.path.isdir(SYN_MODELS_DIRT):    
    synthesizers =  Enum('synthesizers', list((file.name, file) for file in Path(SYN_MODELS_DIRT).glob("**/*.pt")))
    print("Loaded synthesizer models: " + str(len(synthesizers)))
else:
    raise Exception(f"Model folder {SYN_MODELS_DIRT} doesn't exist.")

if os.path.isdir(ENC_MODELS_DIRT):    
    encoders =  Enum('encoders', list((file.name, file) for file in Path(ENC_MODELS_DIRT).glob("**/*.pt")))
    print("Loaded encoders models: " + str(len(encoders)))
else:
    raise Exception(f"Model folder {ENC_MODELS_DIRT} doesn't exist.")

class Model(str, Enum):
    DEFAULT = "default"

class Input(BaseModel):
    model: Model = Field(
        Model.DEFAULT, title="模型类型",
    )
    # datasets_root: str = Field(
    #     ..., alias="预处理数据根目录", description="输入目录（相对/绝对）,不适用于ppg2mel模型",
    #     format=True,
    #     example="..\\trainning_data\\"
    # )
    input_root: str = Field(
        ..., alias="输入目录", description="预处理数据根目录",
        format=True,
        example=f"..{os.sep}audiodata{os.sep}SV2TTS{os.sep}synthesizer"
    )
    run_id: str = Field(
        "", alias="新模型名/运行ID", description="使用新ID进行重新训练，否则选择下面的模型进行继续训练",
    )
    synthesizer: synthesizers = Field(
        ..., alias="已有合成模型", 
        description="选择语音合成模型文件."
    )
    gpu: bool = Field(
        True, alias="GPU训练", description="选择“是”，则使用GPU训练",
    )
    verbose: bool = Field(
        True, alias="打印详情", description="选择“是”，输出更多详情",
    )
    encoder: encoders = Field(
        ..., alias="语音编码模型", 
        description="选择语音编码模型文件."
    )
    save_every: int = Field(
        1000, alias="更新间隔", description="每隔n步则更新一次模型",
    )
    backup_every: int = Field(
        10000, alias="保存间隔", description="每隔n步则保存一次模型",
    )
    log_every: int = Field(
        500, alias="打印间隔", description="每隔n步则打印一次训练统计",
    )

class AudioEntity(BaseModel):
    content: bytes
    mel: Any

class Output(BaseModel):
    __root__: int

    def render_output_ui(self, streamlit_app) -> None:  # type: ignore
        """Custom output UI.
        If this method is implmeneted, it will be used instead of the default Output UI renderer.
        """
        streamlit_app.subheader(f"Training started with code: {self.__root__}")

def train(input: Input) -> Output:
    """Train(训练)"""

    print(">>> Start training ...")
    force_restart = len(input.run_id) > 0
    if not force_restart:
        input.run_id = Path(input.synthesizer.value).name.split('.')[0]
    
    synt_train(
        input.run_id, 
        input.input_root, 
        f"synthesizer{os.sep}saved_models", 
        input.save_every, 
        input.backup_every, 
        input.log_every, 
        force_restart,
        hparams
    )
    return Output(__root__=0)