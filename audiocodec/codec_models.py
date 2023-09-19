import dac
import torch
import sys
import os
import git
import requests
import zipfile
from tqdm import tqdm
from pathlib import Path
from transformers import EncodecModel

class CodecModel:
    def __init__(self):
        self.model = None

    def load_model(self):
        pass
    
    @torch.no_grad()
    def encode_tensor(self, x):
        """
        Args:
            "x" : Tensor[B x 1 x T_wav]

        Returns:
            "codes" : Tensor[B x n_q x T_code]
                Codebook indices for each codebook
                (quantized discrete representation of input)
        """
        pass
    
    @torch.no_grad()
    def decode_tensor(self):
        """
        Args:
            codes (Tensor): Encoded codes to be decoded.

        Returns:
            Tensor: Decoded audio waveform.
        """
        pass

    @property
    def codebook_size(self):
        pass

    @property
    def sample_rate(self):
        pass

    @property
    def downsample_rate(self):
        """
        Sometimes called `stride factor` and `hop size`.
        T_audio = downsample_rate * T_code
        """
        pass

class DAC(CodecModel):
    def __init__(self, model_sr, device):
        super().__init__()
        self.load_model(model_sr, device)
        
    def load_model(self, model_sr, device):
        model_path = dac.utils.download(model_type=model_sr)
        self.model = dac.DAC.load(model_path).to(device)
    
    @torch.no_grad()
    def encode_tensor(self, x: torch.Tensor, padding_mask=None, n_q=None) -> torch.Tensor:
        zq, codes, _, _, _ = self.model.encode(x)
        if n_q is not None:
            codes = codes[:, :n_q, :]
        return codes
    
    @torch.no_grad()
    def decode_tensor(self, codes: torch.Tensor):
        zq, _, codes = self.model.quantizer.from_codes(codes)
        y = self.model.decode(zq)
        return y

    @property
    def codebook_size(self):
        return self.model.codebook_size
    
    @property
    def downsample_rate(self):
        return 320

    @property
    def sample_rate(self):
        return self.model.sample_rate

class Encodec(CodecModel):
    def __init__(self, model_sr, device):
        super().__init__()
        self.load_model(model_sr, device)
        self.nq_bw = {2: 1.5, 4: 3.0, 8: 6.0, 16: 12.0, 32: 24.0}

    def load_model(self, model_sr, device):
        model = EncodecModel.from_pretrained(f"facebook/encodec_{model_sr}")
        self.model = model.to(device)

    @torch.no_grad()
    def encode_tensor(self, x: torch.Tensor, padding_mask=None, n_q=None) -> torch.Tensor:
        bw = self.nq_bw[n_q] if n_q is not None else None
        model_output = self.model.encode(x, padding_mask=padding_mask, bandwidth=bw)
        codes = model_output['audio_codes'].squeeze(0)
        # B x n_q x T
        return codes
    
    @torch.no_grad()
    def decode_tensor(self, codes: torch.Tensor):
        codes = codes.unsqueeze(0) # B x n_q x T -> 1 x B x n_q x T
        y = self.model.decode(codes, audio_scales=[None])['audio_values']
        return y

    @property
    def codebook_size(self):
        return self.model.config.codebook_size
    
    @property
    def downsample_rate(self):
        return 320

    @property
    def sample_rate(self):
        return self.model.config.sampling_rate

class AudioDec(CodecModel):
    def __init__(self, model_sr, device):
        super().__init__()
        self.repo_path = Path(__file__).parent /  "AudioDec"
        sys.path.append(str(self.repo_path))
        self.load_model(model_sr, device)      
        
    def load_model(self, model_sr, device):
        """
        Load the AudioDec model.

        1. Initialize the AudioDec Git submodule if not already initialized.
        2. Download pre-trained model files if not present.
        3. Load the selected model based on sample rate.
        """
        # init git module
        cur_repo = git.Repo(".")
        sm = cur_repo.submodules[0]
        if not sm.module_exists():
            print("[INFO] Init AudioDec repo")
            cur_repo.git.submodule('update', '--init')

        # download exp dir
        exp_dir = self.repo_path / "exp"
        if len(list(exp_dir.rglob('*.pkl'))) == 0:
            url = "https://github.com/facebookresearch/AudioDec/releases/download/pretrain_models/exp.zip"
            zip_file_name = "exp.zip"
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading exp dir from AudioDec repo")
            
            with open(zip_file_name, 'wb') as zip_file:
                for data in response.iter_content(chunk_size=1024):
                    progress_bar.update(len(data))
                    zip_file.write(data)
            
            # Extract the ZIP file
            with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
                zip_ref.extractall(exp_dir.parent)
            os.remove(zip_file_name)

        from .AudioDec.utils.audiodec import AudioDec as AudioDecModel, assign_model
        if model_sr == "24khz":
            self.sr, encoder_checkpoint, decoder_checkpoint = assign_model('libritts_v1')
        elif model_sr == "48khz":
            self.sr, encoder_checkpoint, decoder_checkpoint = assign_model('vctk_v1')
            
        os.chdir(self.repo_path) 
        audiodec = AudioDecModel(tx_device=device, rx_device=device)
        audiodec.load_transmitter(encoder_checkpoint)
        audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)
        self.model = audiodec
        os.chdir(self.repo_path.parent.parent)

    @torch.no_grad()
    def encode_tensor(self, x: torch.Tensor, padding_mask=None, n_q=None) -> torch.Tensor:
        """
        return:
        z: B x D x T
        idx: nq x T
        zq: B x T x D
        """
        self.model.tx_encoder.reset_buffer()
        z = self.model.tx_encoder.encode(x)

        # see Quantizer.encode
        zq, codes = self.model.tx_encoder.quantizer.codebook.forward_index(z.transpose(2, 1), flatten_idx=False)
        if len(codes.shape) == 2:
            # forward index do squeeze batch size.
            # To match our interface, we unsqueeze it if the batch size is 1
            codes = codes.unsqueeze(1)
        codes = codes.transpose(0, 1)
        return codes
    
    @torch.no_grad()
    def decode_tensor(self, codes: torch.Tensor):
        codes = codes.squeeze(0)
        zq = self.model.rx_encoder.lookup(codes)
        y = self.model.decoder.decode(zq)
        # y = y.squeeze(1).transpose(1, 0).cpu().detach().numpy() # (T, C)
        return y

    @property
    def codebook_size(self):
        return self.model.rx_encoder.quantizer.codebook.codebook_size
    
    @property
    def downsample_rate(self):
        # Warning for future compatibility
        return 300

    @property
    def sample_rate(self):
        return self.sr