# 🚀 xLSTM-LLM Pipeline Assets

![GitHub last commit](https://img.shields.io/github/last-commit/UYousafzai/llm_xlstm)
![GitHub issues](https://img.shields.io/github/issues/UYousafzai/llm_xlstm)
![GitHub stars](https://img.shields.io/github/stars/UYousafzai/llm_xlstm)

This repository will contain the essential assets and components for building xLSTM-based (Extended Long Short-Term Memory) Large Language Model (LLM) pipelines. The goal is to provide a comprehensive toolkit for researchers and developers working on advanced natural language processing tasks.

The base code for this project is derived from the [PyxLSTM repository](https://github.com/muditbhargava66/PyxLSTM). I am going to build upon this foundation to create a more comprehensive toolkit for xLSTM-based LLM pipelines. More downstream tasks maybe introduced in the future.

## 📚 Table of Contents

- [🎯 Features](#-features)
- [🛠️ Installation](#️-installation)
- [🚀 Quick Start](#-quick-start)
- [📖 Documentation](#-documentation)
- [🔍 Known Issues and Fixes](#-known-issues-and-fixes)
- [🙏 Acknowledgments](#-acknowledgments)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🎯 Aimed Features

- 🧠 Pretrained Models as well as Public Dataset Loaders
- 🔧 Customizable training scripts for fine-tuning on specific downstream tasks
- 📊 Data preprocessing tools and utilities
- 🔍 Evaluation metrics and benchmarking tools

## 🛠️ Current Installation Guidelines

```bash
git clone https://github.com/UYousafzai/llm_xlstm
cd llm_xlstm
pip install -r requirements.txt
```

## 🚀 Quick Start

```
Details to training your own LLM's with xlstms will be written in future.
```

## 📖 Documentation

Documentation link will be posted here for now its just a link to the repo [documentation](https://github.com/UYousafzai/llm_xlstm).

## 🔍 Read this if you're coming from PyxLSTM repository

We've identified and addressed some implementation issues in the above mentioned PyxLSTM codebase, additionally I aim to introduce the stabalization techniques mentioned by the paper, for now:

1. sLSTM Cell: I have fixed implementation issues with the sLSTM cell. The current version in this repository should work correctly.
2. mLSTM Cell: Untouched from the original repo, will look into it if anything is out of place from the research publication.

## 🙏 Acknowledgments

This project builds upon the work done in the [PyxLSTM repository](https://github.com/muditbhargava66/PyxLSTM) despite having a good foundation, Our company need this repo for standardizing some training procedures for our company's internal consumption, we've decided to Open Source most of these to the public in this repo.

## 🤝 Contributing

Currently I have no plans on how to manage contributions but will look into it if there is enough interest.


## 📄 License

This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE - see the [LICENSE](LICENSE) file for details.