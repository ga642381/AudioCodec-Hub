
import torch
import json
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from .dataset import SpeechDataset
from .codec_models import DAC, Encodec, AudioDec

class AudioCodec:
    def __init__(self, name):
        """
        Model Specifications:
        ----------------------

        Encodec (encodec_24khz):
        +-------+----+---+---+----+----+
        | n_q   |  2 |  4|  8| 16 | 32 |
        +-------+----+---+---+----+----+
        | kbps  | 1.5| 3 | 6 | 12 | 24 |
        +-------+----+---+---+----+----+

        DAC (dac_24khz):
        +-------+----+---+---+----+----+
        | n_q   |  2 |  4|  8| 16 | 32 |
        +-------+----+---+---+----+----+
        | kbps  | 1.5| 3 | 6 | 12 | 24 |
        +-------+----+---+---+----+----+

        AudioDec (audiodec_24khz, audiodec_48khz):
        +-------+----+
        | n_q   |  8 |
        +-------+----+
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_codec_model(name, self.device)
        self.model_type = name.split("_")[0]
        assert self.model_type in ["encodec", "dac", "audiodec"]

    def encode_file(self, in_file, out_file, n_q=None, codebook_offset=False):
        """
        Encode the audio file into codes and save the resulting codes to an output file.

        Args:
            in_file (str): Path to the input audio file.
            out_file (str): Path to the output file where encoded codes will be saved.
            n_q (int, optional): Number of codes to select from the encoded codes. If provided, only the first
                'n_q' codes will be kept. Default is None, meaning all codes will be kept.
            codebook_offset (bool, optional): Whether to apply a codebook offset. Default is False.
        """
        x, _ = librosa.load(in_file, sr=self.model.sample_rate)
        x = torch.tensor(x).to(self.device).unsqueeze(0).unsqueeze(0) # [1, 1, L]
        codes = self.model.encode_tensor(x, n_q=n_q)

        # wrapup codes
        codes = codes.cpu().squeeze(0).numpy()

        # save codes
        self._save_file(codes, out_file, codebook_offset)

    def encode_dir(self, in_dir, out_dir, n_q=None, batch_size=2, codebook_offset=False):
        """
        Encode audio files from the input directory and save the encoded files to the output directory.

        Args:
            in_dir (str or Path): The input directory containing the audio files to be encoded.
            out_dir (str or Path): The output directory where the encoded files will be saved.
        """
        in_dir = Path(in_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        speech_dataloader = self._load_dataloader(in_dir, batch_size)
        for speech_batch, masks_batch, file_paths in tqdm(speech_dataloader):
            speech_batch = speech_batch.to(self.device)
            codes_batch = self.model.encode_tensor(speech_batch, masks_batch, n_q=n_q)

            codes_list= self._wrapup_codes(codes_batch, masks_batch)
            for codes, file_path in zip(codes_list, file_paths):
                out_file = out_dir / f"{file_path.stem}.json"

                codes = codes.cpu().squeeze(0).numpy()
                self._save_file(codes, out_file, codebook_offset)

    def decode_file(self, in_file, out_file, codebook_offset):
        """
        Decode codes from an input file and save the audio waveform to an output file.

        Args:
            in_file (str): Path to the input file containing encoded codes.
            out_file (str): Path to the output file where the decoded audio waveform will be saved.
            codebook_offset (bool): Whether a codebook offset was applied during encoding.
        """
        codes = self._load_file(in_file, codebook_offset)
        codes = torch.tensor(codes).unsqueeze(0).to(self.device)
        wav = self.model.decode_tensor(codes)
        wav = wav.cpu().numpy().squeeze(0).squeeze(0)
        sf.write(out_file, wav, self.model.sample_rate, "PCM_16")

    def decode_dir(self, in_dir, out_dir, codebook_offset=False):
        """
        Decode codes from input files in a directory and save the audio waveforms to output files.

        Args:
            in_dir (str): Path to the directory containing encoded code files.
            out_dir (str): Path to the directory where the decoded audio waveforms will be saved.
            codebook_offset (bool): Whether a codebook offset was applied during encoding.
        """
        in_dir = Path(in_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for in_file in in_dir.glob("*.json"):
            out_file = out_dir / in_file.with_suffix(".wav").name
            self.decode_file(in_file, out_file, codebook_offset)

    def _load_codec_model(self, name, device):
        model_type, model_sr = name.split("_")
        assert model_type in ['dac', 'encodec', 'audiodec']
        assert model_sr in ['24khz']

        if model_type == "dac":
            model = DAC(model_sr, device)
        elif model_type == "encodec":
            model = Encodec(model_sr, device)
        elif model_type == 'audiodec':
            model = AudioDec(model_sr, device)
        return model
    

    def _load_dataloader(self, data_dir, batch_size):
        """
        Load and return a dataloader for a given dataset directory.

        Args:
            data_dir (str): Path to the directory containing audio waveform files.
            batch_size (int): Batch size for the dataloader.

        Returns:
            DataLoader: The dataloader for the given dataset.
        """
        wav_fns = [Path(x) for x in librosa.util.find_files(data_dir)]
        speech_dataset = SpeechDataset(wav_fns, target_sr=self.model.sample_rate)
        dataloader = DataLoader(speech_dataset, batch_size=batch_size, shuffle=False, collate_fn=speech_dataset.collate_fn)
        print(f"Number of wav files found: {len(speech_dataset)}")
        return dataloader

    def _save_file(self, codes, out_file, codebook_offset):
        """
        Save a numpy array of codes to a file in JSON format.

        Parameters:
            codes (np array): Numpy array of codes corresponding to one audio file.
            out_file (str): Output file path.
            codebook_offset (bool): Whether to apply codebook offset.

        The function shifts codes if codebook_offset is True, converts the codes to a dictionary,
        and writes the dictionary to the specified JSON file.
        """
        # shift
        if codebook_offset:
            codes = self._shift_codes(codes)

        # make dict
        codes_dict = {}
        for i , codes_i in enumerate(codes):
            codes_i = [str(c) for c in codes_i]
            codes_dict[str(i)] = " ".join(codes_i)

        # write
        with open(out_file, "w") as f:
            json.dump(codes_dict, f, indent=4)

    def _load_file(self, in_file, codebook_offset):
        """
        Load encoded codes from an input file and optionally apply a codebook offset.

        Args:
            in_file (str): Path to the input file containing encoded codes in JSON format.
            codebook_offset (bool): Whether to apply a codebook offset to the loaded codes.

        Returns:
            np.ndarray: Loaded encoded codes as a NumPy array.
        """
        with open(in_file, "r") as f:
            codes_dict = json.load(f)
            codes = []
            for i in range(len(codes_dict)):
                codes_i_str = codes_dict[str(i)]
                codes_i = np.array([int(c) for c in codes_i_str.split()])
                codes.append(codes_i)
            codes = np.stack(codes)
        

        if self.model_type == "audiodec":
            # These kinds of models expect the codes are offset
            if not codebook_offset:
                codes = self._shift_codes(codes, "add")
        elif self.model_type in ["encodec", "dac"]:
            # These kinds of models expect the codes not offset
            if codebook_offset:
                codes = self._shift_codes(codes, "sub")
        else:
            raise NotImplementedError
        
        return codes
    
    def _wrapup_codes(self, codes_batch, masking_batch):
        """
        Process encoded codes and apply masking.
        
        Parameters:
            codes (np array): Encoded codes of shape B x n_q x T_code.
            masking: Masking array of shape B x T_audio.
            
        Returns:
            List of numpy arrays, each of shape n_q x T_code.
        """
        codes_list = []
        for codes, masks in zip(codes_batch, masking_batch):
            c_len = len(masks.nonzero()) // self.model.downsample_rate
            real_codes = codes[:, :c_len]
            codes_list.append(real_codes)
        return codes_list
    
    def _shift_codes(self, codes, mode="add"):
        """
        codes: np array of codes corresponding to one audio
        """
        for i in range(len(codes)):
            offset_const = int(i * self.model.codebook_size)
            if mode == "add":
                codes[i] += offset_const
            elif mode == "sub":
                codes[i] -= offset_const
        return codes