# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from setuptools import setup, find_packages

setup(
    name="hy3dgen",
    version="2.0.2",
    description="HiCache Hermite acceleration for Tencent Hunyuan3D-2 shape generation.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Archerkattri/hunyuan2-plus",
    project_urls={
        "Upstream": "https://github.com/Tencent/Hunyuan3D-2",
        "Repository": "https://github.com/Archerkattri/hunyuan2-plus",
    },
    license="LicenseRef-Tencent-Hunyuan-3D-2.0-Community",
    license_files=["LICENSE", "NOTICE"],
    python_requires=">=3.11",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    packages=find_packages(),
    include_package_data=True,
    package_data={"hy3dgen": ["assets/*", "assets/**/*"]},
    install_requires=[
        'gradio',
        "tqdm>=4.66.3",
        'numpy',
        'ninja',
        'diffusers',
        'hicache-pp>=1.2.1',
        'pybind11',
        'opencv-python',
        'einops',
        "transformers>=4.48.0",
        'omegaconf',
        'trimesh',
        'pymeshlab',
        'pygltflib',
        'xatlas',
        'accelerate',
        'gradio',
        'fastapi',
        'uvicorn',
        'rembg',
        'onnxruntime'
    ]
)
