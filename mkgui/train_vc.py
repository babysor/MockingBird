from pydantic import BaseModel, Field
import os
from pathlib import Path
from enum import Enum
from typing import Any, Tuple
import numpy as np
from utils.load_yaml import HpsYaml
from utils.util import AttrDict
import torch

# Constants
EXT_MODELS_DIRT = f"ppg_extractor{os.sep}saved_models"
CONV_MODELS_DIRT = f"ppg2mel{os.sep}saved_models"
ENC_MODELS_DIRT = f"encoder{os.sep}saved_models"


if os.path.isdir(EXT_MODELS_DIRT):    
    extractors =  Enum('extractors', list((file.name, file) for file in Path(EXT_MODELS_DIRT).glob("**/*.pt")))
    print("Loaded extractor models: " + str(len(extractors)))
else:
    raise Exception(f"Model folder {EXT_MODELS_DIRT} doesn't exist.")

if os.path.isdir(CONV_MODELS_DIRT):    
    convertors =  Enum('convertors', list((file.name, file) for file in Path(CONV_MODELS_DIRT).glob("**/*.pth")))
    print("Loaded convertor models: " + str(len(convertors)))
else:
    raise Exception(f"Model folder {CONV_MODELS_DIRT} doesn't exist.")

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
        Model.VC_PPG2MEL, title="模型类型",
    )
    # datasets_root: str = Field(
    #     ..., alias="预处理数据根目录", description="输入目录（相对/绝对）,不适用于ppg2mel模型",
    #     format=True,
    #     example="..\\trainning_data\\"
    # )
    output_root: str = Field(
        ..., alias="输出目录(可选)", description="建议不填，保持默认",
        format=True,
        example=""
    )
    continue_mode: bool = Field(
        True, alias="继续训练模式", description="选择“是”，则从下面选择的模型中继续训练",
    )
    gpu: bool = Field(
        True, alias="GPU训练", description="选择“是”，则使用GPU训练",
    )
    verbose: bool = Field(
        True, alias="打印详情", description="选择“是”，输出更多详情",
    )
    # TODO: Move to hiden fields by default
    convertor: convertors = Field(
        ..., alias="转换模型", 
        description="选择语音转换模型文件."
    )
    extractor: extractors = Field(
        ..., alias="特征提取模型", 
        description="选择PPG特征提取模型文件."
    )
    encoder: encoders = Field(
        ..., alias="语音编码模型", 
        description="选择语音编码模型文件."
    )
    njobs: int = Field(
        8, alias="进程数", description="适用于ppg2mel",
    )
    seed: int = Field(
        default=0, alias="初始随机数", description="适用于ppg2mel",
    )
    model_name: str = Field(
        ..., alias="新模型名", description="仅在重新训练时生效,选中继续训练时无效",
        example="test"
    )
    model_config: str = Field(
        ..., alias="新模型配置", description="仅在重新训练时生效,选中继续训练时无效",
        example=".\\ppg2mel\\saved_models\\seq2seq_mol_ppg2mel_vctk_libri_oneshotvc_r4_normMel_v2"
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

def train_vc(input: Input) -> Output:
    """Train VC(训练 VC)"""

    print(">>> OneShot VC training ...")
    params = AttrDict()
    params.update({
        "gpu": input.gpu,
        "cpu": not input.gpu,
        "njobs": input.njobs,
        "seed": input.seed,
        "verbose": input.verbose,
        "load": input.convertor.value,
        "warm_start": False,
    })
    if input.continue_mode: 
        # trace old model and config
        p = Path(input.convertor.value)
        params.name = p.parent.name
        # search a config file
        model_config_fpaths = list(p.parent.rglob("*.yaml"))
        if len(model_config_fpaths) == 0:
            raise "No model yaml config found for convertor"
        config = HpsYaml(model_config_fpaths[0])
        params.ckpdir = p.parent.parent
        params.config = model_config_fpaths[0]
        params.logdir = os.path.join(p.parent, "log")
    else:
        # Make the config dict dot visitable
        config = HpsYaml(input.config)    
    np.random.seed(input.seed)
    torch.manual_seed(input.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(input.seed)
    mode = "train"
    from ppg2mel.train.train_linglf02mel_seq2seq_oneshotvc import Solver
    solver = Solver(config, params, mode)
    solver.load_data()
    solver.set_model()
    solver.exec()
    print(">>> Oneshot VC train finished!")

    # TODO: pass useful return code
    return Output(__root__=(input.dataset, 0))