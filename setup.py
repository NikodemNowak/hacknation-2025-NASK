from setuptools import setup, find_packages

setup(
    name="pllum_anonymizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.36.0",
        "torch>=2.0.0",
        "langchain-openai>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    python_requires=">=3.9",
    author="all_in()",
    author_email="",
    description="Narzędzie do anonimizacji danych dla modelu PLLUM - Dane bez twarzy",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/NikodemNowak/hacknation-2025-NASK",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Natural Language :: Polish",
    ],
    keywords="anonymization, nlp, polish, pii, data-privacy, pllum",
)