import unittest
from audiocodec.audio_codec import AudioCodec
import os
import shutil
from pathlib import Path
from itertools import product

def test_model(model, model_name, test_results_dir, test_wav_file, test_wav_dir, n_q, codebook_offset):
    model_test_results_dir = os.path.join(test_results_dir)
    
    test_code = Path(f"{model_test_results_dir}/test_codes/nq_{n_q}_offset_{codebook_offset}/121_121726_000000_000000.json")
    test_rewav = Path(f"{model_test_results_dir}/test_rewavs/nq_{n_q}_offset_{codebook_offset}/121_121726_000000_000000.wav")

    for path in [test_code.parent, test_rewav.parent]:
        path.mkdir(parents=True, exist_ok=True)

    print(f"Testing with n_q = {n_q} and codebook_offset = {codebook_offset}")
    
    # Test encode_file and decode_file for the given model
    print(f"\nTesting encode_file and decode_file with {model_name} model...")
    model.encode_file(test_wav_file, out_file=test_code, n_q=n_q, codebook_offset=codebook_offset)
    model.decode_file(test_code, out_file=test_rewav, codebook_offset=codebook_offset)
    
    # Test encode_dir for the given model
    print(f"\nTesting directory encoding with {model_name} model...")
    test_code_dir = Path(f"{model_test_results_dir}/test_codes/nq_{n_q}_offset_{codebook_offset}")
    model.encode_dir(test_wav_dir, test_code_dir, n_q=n_q, batch_size=2, codebook_offset=codebook_offset)
    
    # Test decode_dir for the given model
    print(f"\nTesting directory decoding with {model_name} model...")
    test_rewav_dir = Path(f"{model_test_results_dir}/test_rewavs/nq_{n_q}_offset_{codebook_offset}")
    model.decode_dir(test_code_dir, test_rewav_dir, codebook_offset=codebook_offset)


class TestDAC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create test directories if they don't exist
        cls.test_results_dir = "./test_results"
        cls.test_wav_file = "./test_wavs/121_121726_000000_000000.wav"
        cls.test_wav_dir = "./test_wavs"

        os.makedirs(cls.test_results_dir, exist_ok=True)

    def setUp(self):
        # Clean up test_results directory before each test
        shutil.rmtree(self.test_results_dir, ignore_errors=True)
                
    def test_dac(self):
        model_dac = AudioCodec(name="dac_24khz")
        codebook_offset_values = [True, False]
        n_q_values = [2, 4, 8, 16, 32]

        test_combinations = product(n_q_values, codebook_offset_values)
        for n_q, codebook_offset in test_combinations:
            with self.subTest(n_q=n_q, codebook_offset=codebook_offset):
                test_model(model_dac, "encodec", self.test_results_dir, self.test_wav_file, self.test_wav_dir, n_q, codebook_offset)

class TestEncodec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create test directories if they don't exist
        cls.test_results_dir = "./test_results"
        cls.test_wav_file = "./test_wavs/121_121726_000000_000000.wav"
        cls.test_wav_dir = "./test_wavs"

        os.makedirs(cls.test_results_dir, exist_ok=True)

    def setUp(self):
        # Clean up test_results directory before each test
        shutil.rmtree(self.test_results_dir, ignore_errors=True)

    def test_encodec(self):
        model_encodec = AudioCodec(name="encodec_24khz")

        codebook_offset_values = [True, False]
        n_q_values = [2, 4, 8, 16, 32]

        test_combinations = product(n_q_values, codebook_offset_values)
        for n_q, codebook_offset in test_combinations:
            with self.subTest(n_q=n_q, codebook_offset=codebook_offset):
                test_model(model_encodec, "encodec", self.test_results_dir, self.test_wav_file, self.test_wav_dir, n_q, codebook_offset)

class TestAudioDec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create test directories if they don't exist
        cls.test_results_dir = "./test_results"
        cls.test_wav_file = "./test_wavs/121_121726_000000_000000.wav"
        cls.test_wav_dir = "./test_wavs"

        os.makedirs(cls.test_results_dir, exist_ok=True)

    def setUp(self):
        # Clean up test_results directory before each test
        shutil.rmtree(self.test_results_dir, ignore_errors=True)
                
    def test_audiodec(self):
        model_audiodec = AudioCodec(name="audiodec_24khz")

        codebook_offset_values = [True, False]
        n_q_values = [8]

        test_combinations = product(n_q_values, codebook_offset_values)
        for n_q, codebook_offset in test_combinations:
            with self.subTest(n_q=n_q, codebook_offset=codebook_offset):
                test_model(model_audiodec, "encodec", self.test_results_dir, self.test_wav_file, self.test_wav_dir, n_q, codebook_offset)

    
if __name__ == "__main__":
    unittest.main()
