from multiprocessing.pool import Pool 

from functools import partial
from itertools import chain
from pathlib import Path
from tqdm import tqdm
import numpy as np
from encoder import inference as encoder
from synthesizer.preprocess_speaker import preprocess_speaker_general
from synthesizer.preprocess_transcript import preprocess_transcript_aishell3, preprocess_transcript_magicdata

data_info = {
    "aidatatang_200zh": {
        "subfolders": ["corpus/train"],
        "trans_filepath": "transcript/aidatatang_200_zh_transcript.txt",
        "speak_func": preprocess_speaker_general
    },
    "magicdata": {
        "subfolders": ["train"],
        "trans_filepath": "train/TRANS.txt",
        "speak_func": preprocess_speaker_general,
        "transcript_func": preprocess_transcript_magicdata,
    },
    "aishell3":{
        "subfolders": ["train/wav"],
        "trans_filepath": "train/content.txt",
        "speak_func": preprocess_speaker_general,
        "transcript_func": preprocess_transcript_aishell3,
    },
    "data_aishell":{
        "subfolders": ["wav/train"],
        "trans_filepath": "transcript/aishell_transcript_v0.8.txt",
        "speak_func": preprocess_speaker_general
    }
}

def preprocess_dataset(datasets_root: Path, out_dir: Path, n_processes: int,
                           skip_existing: bool, hparams, no_alignments: bool,
                           dataset: str):
    dataset_info = data_info[dataset]
    # Gather the input directories
    dataset_root = datasets_root.joinpath(dataset)
    input_dirs = [dataset_root.joinpath(subfolder.strip()) for subfolder in dataset_info["subfolders"]]
    print("\n    ".join(map(str, ["Using data from:"] + input_dirs)))
    assert all(input_dir.exists() for input_dir in input_dirs)
    
    # Create the output directories for each output file type
    out_dir.joinpath("mels").mkdir(exist_ok=True)
    out_dir.joinpath("audio").mkdir(exist_ok=True)
    
    # Create a metadata file
    metadata_fpath = out_dir.joinpath("train.txt")
    metadata_file = metadata_fpath.open("a" if skip_existing else "w", encoding="utf-8")

    # Preprocess the dataset
    dict_info = {}
    transcript_dirs = dataset_root.joinpath(dataset_info["trans_filepath"])
    assert transcript_dirs.exists(), str(transcript_dirs)+" not exist."
    with open(transcript_dirs, "r", encoding="utf-8") as dict_transcript:
        # process with specific function for your dataset 
        if "transcript_func" in dataset_info:
            dataset_info["transcript_func"](dict_info, dict_transcript)
        else:
            for v in dict_transcript:
                if not v:
                    continue
                v = v.strip().replace("\n","").replace("\t"," ").split(" ")
                dict_info[v[0]] = " ".join(v[1:])

    speaker_dirs = list(chain.from_iterable(input_dir.glob("*") for input_dir in input_dirs))
    func = partial(dataset_info["speak_func"], out_dir=out_dir, skip_existing=skip_existing, 
                   hparams=hparams, dict_info=dict_info, no_alignments=no_alignments)
    job = Pool(n_processes).imap(func, speaker_dirs)
    for speaker_metadata in tqdm(job, dataset, len(speaker_dirs), unit="speakers"):
        for metadatum in speaker_metadata:
            metadata_file.write("|".join(str(x) for x in metadatum) + "\n")
    metadata_file.close()

    # Verify the contents of the metadata file
    with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
    mel_frames = sum([int(m[4]) for m in metadata])
    timesteps = sum([int(m[3]) for m in metadata])
    sample_rate = hparams.sample_rate
    hours = (timesteps / sample_rate) / 3600
    print("The dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." %
          (len(metadata), mel_frames, timesteps, hours))
    print("Max input length (text chars): %d" % max(len(m[5]) for m in metadata))
    print("Max mel frames length: %d" % max(int(m[4]) for m in metadata))
    print("Max audio timesteps length: %d" % max(int(m[3]) for m in metadata))

def embed_utterance(fpaths, encoder_model_fpath):
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    # Compute the speaker embedding of the utterance
    wav_fpath, embed_fpath = fpaths
    wav = np.load(wav_fpath)
    wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(wav)
    np.save(embed_fpath, embed, allow_pickle=False)
    
 
def create_embeddings(synthesizer_root: Path, encoder_model_fpath: Path, n_processes: int):
    wav_dir = synthesizer_root.joinpath("audio")
    metadata_fpath = synthesizer_root.joinpath("train.txt")
    assert wav_dir.exists() and metadata_fpath.exists()
    embed_dir = synthesizer_root.joinpath("embeds")
    embed_dir.mkdir(exist_ok=True)
    
    # Gather the input wave filepath and the target output embed filepath
    with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
        fpaths = [(wav_dir.joinpath(m[0]), embed_dir.joinpath(m[2])) for m in metadata]
        
    # TODO: improve on the multiprocessing, it's terrible. Disk I/O is the bottleneck here.
    # Embed the utterances in separate threads
    func = partial(embed_utterance, encoder_model_fpath=encoder_model_fpath)
    job = Pool(n_processes).imap(func, fpaths)
    list(tqdm(job, "Embedding", len(fpaths), unit="utterances"))
