# AudioCodec-Hub

AudioCodec-Hub is a Python library for encoding and decoding audio data, supporting various neural audio codec models. It provides an easy-to-use interface for encoding audio files and directories in batch mode, which is useful when conducting research on large speech language models.

## Supported Models

| Model Specification ID | Model        | Codebook Sizes  | Bitrates (kbps)    | Downsample Rate | Code Rate |
|------------------------|--------------|-----------------|--------------------|-----------------|-----------|
| encodec_24khz          | EnCodec      | 2, 4, 8, 16, 32 | 1.5, 3, 6, 12, 24  |       320       |     75 Hz |
| dac_24khz              | DAC          | 2, 4, 8, 16, 32 | 1.5, 3, 6, 12, 24  |       320       |     75 Hz  |
| audiodec_24khz         | AudioDec     | 8               | 6.4                |       300       |     80 Hz |
| audiodec_48khz         | AudioDec     | 8               | 12.8               |       300       |     160 Hz|

- EnCodec: High Fidelity Neural Audio Compression [[GitHub](https://github.com/facebookresearch/encodec)] [[Paper](https://arxiv.org/abs/2210.13438)]
- DAC: High-Fidelity Audio Compression with Improved RVQGAN [[GitHub](https://github.com/descriptinc/descript-audio-codec)] [[Paper](https://arxiv.org/abs/2306.06546)]
- AudioDec: An Open-source Streaming High-fidelity Neural Audio Codec [[GitHub](https://github.com/facebookresearch/AudioDec)] [[Paper](https://arxiv.org/abs/2305.16608)]

PR is welcome to support more settings!

## Features

- **Batch Encoding**: Process multiple audio files in a directory in batch mode. This feature is particularly useful when you want to train large speech models.
- **Codebook Offset**: Apply codebook offsets to encoded data as an option.
- **Custom Models**: Implement your own custom models with ease. See the [custom models](#custom-models) section.

## To-Do List:
- [ ] Implement batch decoding of audio files.
- [ ] Support handling multiple channels within a single audio file.
- [ ] Support other settings in encodec.
- [ ] Support other settings in dac.
- [ ] Provide a Colab demo.
- [ ] Make it a PyPI pacakge. 

## Install
```bash
pip install git+https://github.com/ga642381/AudioCodec-Hub.git
```

## Usage
### Encoding and Decoding One Sinfle File
Here's an example of how to use AudioCodec-Hub to encode and decode one single file:

```python
from audiocodec import AudioCodec

NQ = 8
CODEBOOK_OFFSET = True

# Initialize the audio codec with a specific model name
model_name = "encodec_24khz"
audio_codec = AudioCodec(model_name)

# Encode an audio file
f_in_enc = "test_wavs/61_70970_000007_000001.wav"
f_out_enc = "encoded.json"
audio_codec.encode_file(f_in_enc, f_out_enc, n_q=NQ, codebook_offset=CODEBOOK_OFFSET)

# Decode the encoded audio data
f_out_dec = "decoded.wav"
audio_codec.decode_file(f_out_enc, f_out_dec, codebook_offset=CODEBOOK_OFFSET)

print("Encoding and decoding completed successfully!")
```

### Encoding and Decoding an Entire Directory in Batch Mode
```python
from audiocodec import AudioCodec

NQ = 8
CODEBOOK_OFFSET = True
BATCH_SIZE = 8

# Initialize the audio codec with a specific model name
model_name = "encodec_24khz"
audio_codec = AudioCodec(model_name)

# Encode all audio files in a directory (Support batch mode)
dir_in_enc = "test_wavs"
dir_out_enc = "encoded_dir"
audio_codec.encode_dir(dir_in_enc, dir_out_enc, n_q=NQ, codebook_offset=CODEBOOK_OFFSET, batch_size=BATCH_SIZE)

# Decode all encoded audio files in a directory (Currently not supporting batch mode)
dir_out_dec = "decoded_dir"
audio_codec.decode_dir(dir_out_enc, dir_out_dec, codebook_offset=CODEBOOK_OFFSET)

print("Encoding and decoding completed successfully!")
```

## Custom Models

AudioCodec-Hub allows you to implement custom audio codec models seamlessly. To create your own codec model, you only need to define a class that inherits from `CodecModel` and implement a few key functions. Contributions to the project through pull requests are always welcome!

### Implementing a Custom Model

1. Begin by creating a Python class that inherits from `CodecModel`. Here's a basic template to get you started:

```python
from audiocodechub import CodecModel
import torch

class CustomAudioCodec(CodecModel):
    def __init__(self):
        super().__init__()
        # Initialize your custom model here

    def load_model(self):
        # Load your custom model from a file or initialize it here
        pass

    @torch.no_grad()
    def encode_tensor(self, x):
        """
        Implement your encoding logic here.
        Args:
            x (Tensor): Input audio tensor [B x 1 x T_wav].

        Returns:
            codes (Tensor): Encoded codes [B x n_q x T_code].
        """
        pass

    @torch.no_grad()
    def decode_tensor(self, codes):
        """
        Implement your decoding logic here.
        Args:
            codes (Tensor): Encoded codes to be decoded.

        Returns:
            Tensor: Decoded audio waveform.
        """
        pass

    @property
    def codebook_size(self):
        """
        Define the size of your custom model's codebook.
        """
        pass

    @property
    def sample_rate(self):
        """
        Define the sample rate of your custom model.
        """
        pass

    @property
    def downsample_rate(self):
        """
        Define the downsampling rate of your custom model.
        """
        pass
```

2. In your custom class, implement the `encode_tensor` and `decode_tensor` functions according to your model's encoding and decoding logic.

3. Set the properties `codebook_size`, `sample_rate`, and `downsample_rate` with the appropriate values for your custom model.

## Unittesting
```
python -m unittest discover -s audiocodec/tests
```

## Disclaimer
* I haven't carefully checked the encoded codes in batch mode. There might be minor mismatches when using these pre-trained models. However, I have provided unit tests, and the resynthesis results can be found in the **test_results/** directory.

* This package serves as a wrapper for neural audio/speech codec models. I have used this package for conducting experiments, but it's important to note that all credit for the pre-trained models goes to their respective creators, not me. I simply provide a wrapper and create a unified interface, which can be still futher improved. I hope you find this package useful.


## Contributions

We encourage contributions to AudioCodec-Hub, including the addition of new custom models. If you've implemented a new model, feel free to submit a pull request to include it in the project. Your contributions are greatly appreciated!
