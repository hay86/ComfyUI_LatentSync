import os
import torch
import random
import torchaudio
import folder_paths
import numpy as np


def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir


class LatentSyncNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "video_path": ("STRING", {"multiline": False, }),
                    "audio": ("AUDIO", ),
                    "seed" :("INT",{"default": 1247}),
                     },}

    CATEGORY = "LatentSyncNode"

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("video_path", )
    FUNCTION = "inference"


    def inference(self, video_path, audio, seed):
        cur_dir = get_ext_dir()
        #ckpt_dir = os.path.join(folder_paths.models_dir, "latentsync")
        ckpt_dir = os.path.join(cur_dir, "checkpoints")
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(ckpt_dir):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="chunyu-li/LatentSync", 
                              allow_patterns=["latentsync_unet.pt", "whisper/tiny.pt"],
                              local_dir=ckpt_dir, local_dir_use_symlinks=False)
            
        infer_py = os.path.join(cur_dir, "scripts/inference.py")
        unet_config_path = os.path.join(cur_dir, "configs/unet/second_stage.yaml")
        scheduler_config_path = os.path.join(cur_dir, "configs")
        ckpt_path = os.path.join(ckpt_dir, "latentsync_unet.pt")
        whisper_ckpt_path = os.path.join(ckpt_dir, "whisper/tiny.pt")

        output_name = ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
        output_video_path = os.path.join(output_dir, f"latentsync_{output_name}_out.mp4")

        # resample audio to 16k hz and save to wav
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        if waveform.dim() == 3: # Expected shape: [channels, samples]
            waveform = waveform.squeeze(0)

        if sample_rate != 16000:
            new_sample_rate = 16000
            waveform_16k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)(waveform)
            waveform, sample_rate = waveform_16k, new_sample_rate

        audio_path = os.path.join(output_dir, f"latentsync_{output_name}_audio.wav")
        torchaudio.save(audio_path, waveform, sample_rate)

        assert os.path.exists(video_path), "video_path not exists"
        assert os.path.exists(audio_path), "audio_path not exists"

        # inference
        env = ':'.join([os.environ.get('PYTHONPATH', ''), cur_dir])
        cmd = f"""PYTHONPATH={env} python {infer_py} --unet_config_path "{unet_config_path}" --inference_ckpt_path "{ckpt_path}" --video_path "{video_path}" --audio_path "{audio_path}" --video_out_path {output_video_path} --seed {seed} --scheduler_config_path {scheduler_config_path} --whisper_ckpt_path {whisper_ckpt_path} """
        
        print(cmd)
        os.system(cmd)

        return (output_video_path,)


NODE_CLASS_MAPPINGS = {
    "D_LatentSyncNode": LatentSyncNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "D_LatentSyncNode": "LatentSync Node",
}