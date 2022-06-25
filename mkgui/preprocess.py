from pydantic import BaseModel, Field
import os
from pathlib import Path
from enum import Enum
from typing import Any, Tuple


# Constants
EXT_MODELS_DIRT = f"ppg_extractor{os.sep}saved_models"
ENC_MODELS_DIRT = f"encoder{os.sep}saved_models"


if os.path.isdir(EXT_MODELS_DIRT):    
    extractors =  Enum('extractors', list((file.name, file) for file in Path(EXT_MODELS_DIRT).glob("**/*.pt")))
    print("Loaded extractor models: " + str(len(extractors)))
else:
    raise Exception(f"Model folder {EXT_MODELS_DIRT} doesn't exist.")

if os.path.isdir(ENC_MODELS_DIRT):    
    encoders = Enum('encoders', list((file.name, file) for file in Path(ENC_MODELS_DIRT).glob("**/*.pt")))
    print("Loaded encoders models: " + str(len(encoders)))
else:
    raise Exception(f"Model folder {ENC_MODELS_DIRT} doesn't exist.")

class Model(str, Enum):
    VC_PPG2MEL = "ppg2mel"

class Dataset(str, Enum):
    AIDATATANG_200ZH = "aidatatang_200zh"
    AIDATATANG_200ZH_S = "aidatatang_200zh_s"

class Input(BaseModel):
    # def render_input_ui(st, input) -> Dict: 
    #     input["selected_dataset"] = st.selectbox(
    #         '选择数据集', 
    #         ("aidatatang_200zh", "aidatatang_200zh_s")
    #     )
    # return input
    model: Model = Field(
        Model.VC_PPG2MEL, title="目标模型",
    )
    dataset: Dataset = Field(
        Dataset.AIDATATANG_200ZH, title="数据集选择",
    )
    datasets_root: str = Field(
        ..., alias="数据集根目录", description="输入数据集根目录（相对/绝对）",
        format=True,
        example="..\\trainning_data\\"
    )
    output_root: str = Field(
        ..., alias="输出根目录", description="输出结果根目录（相对/绝对）",
        format=True,
        example="..\\trainning_data\\"
    )
    n_processes: int = Field(   
        2, alias="处理线程数", description="根据CPU线程数来设置",
        le=32, ge=1
    )
    extractor: extractors = Field(
        ..., alias="特征提取模型", 
        description="选择PPG特征提取模型文件."
    )
    encoder: encoders = Field(
        ..., alias="语音编码模型", 
        description="选择语音编码模型文件."
    )

class AudioEntity(BaseModel):
    content: bytes
    mel: Any

class Output(BaseModel):
    __root__: Tuple[str, int]

    def render_output_ui(self, streamlit_app, input) -> None:  # type: ignore
        """Custom output UI.
        If this method is implmeneted, it will be used instead of the default Output UI renderer.
        """
        sr, count = self.__root__
        streamlit_app.subheader(f"Dataset {sr} done processed total of {count}")

def preprocess(input: Input) -> Output:
    """Preprocess(预处理)"""
    finished = 0
    if input.model == Model.VC_PPG2MEL:
        from ppg2mel.preprocess import preprocess_dataset
        finished = preprocess_dataset(
            datasets_root=Path(input.datasets_root),
            dataset=input.dataset,
            out_dir=Path(input.output_root),
            n_processes=input.n_processes,
            ppg_encoder_model_fpath=Path(input.extractor.value),
            speaker_encoder_model=Path(input.encoder.value)
        )
    # TODO: pass useful return code
    return Output(__root__=(input.dataset, finished))