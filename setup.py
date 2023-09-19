from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name='audiocodec-hub',
    version='1.2',
    packages=find_packages(),
    author="Kai-Wei Chang",
    author_email="kaiwei.chang.tw@gmail.com",
    description="AudioCodec-Hub is a Python library for encoding and decoding audio data, supporting various neural audio codec models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'GitPython',
        'librosa',
        'numpy',
        'PyYAML',
        'scikit-learn',
        'soundfile',
        'tqdm',
        'transformers==4.33.2',
        'requests',
        'descript-audio-codec==1.0.0',
        'torch',
        'torchaudio'
    ],
)