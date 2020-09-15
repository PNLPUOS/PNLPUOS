# Practical NLP Study Project
Practical NLP for Survey Analysis with deepsight GmbH at Universität Osnabrück.

### Overview

This repository hosts the code for the study project 'Practical NLP', an interdisciplinary study project at Osnabrück University held during WS2019-SS2020. The project implements a topic analysis pipeline including data cleaning, sentiment analysis, and document clustering, optimized to employee survey data. For further information on the project scope, individual modules, and methods employed, please refer to the online [documentation](https://pnlpuos.github.io/). 

***

### Project Structure

<pre>
|-- data
|-- notebooks
    |-- sentiment_analysis
    |-- topic_clustering
    |-- data_prep
|-- src
    |-- sentiment_classifier
    |-- topic_modeling
    |-- outputs
</pre>


***

### Installation

1. Clone the repository.

<pre>$ git clone https://github.com/PNLPUOS/PNLPUOS.git</pre>

2. Install PyTorch from [source](https://pytorch.org/). The package is tested on Python==3.7.4 with pytorch==1.6.0. Please ensure that your Python installation matches your system (32 or 64bit).
3. Navigate to the cloned directory and install with pip. Ensure your environment has Cython installed.

<pre>$ pip install .</pre>

3. Note: OS X users require a compiler with good C++11 support per the [FastText documentation](https://fasttext.cc/docs/en/support.html). Information on how to install one of the available compilers can be found [here](https://www.ics.uci.edu/~pattis/common/handouts/macclion/clang.html).
4. Obtain fasttext English model [here](https://fasttext.cc/docs/en/english-vectors.html). Place ‘common-crawl-300d-2M-subword’ in 'pnlp' directory.

### Usage

4. Obtain a dataset for analysis. The pipeline is optimized to employee survey comments in .csv  (semicolon-delineated) conforming to the following format:

| Report Grouping | Question Text     | Comment   |
| --------------- | ----------------- | --------- |
| Department 1    | Survey Question 1 | Comment 1 |
| Department 2    | Survey Question 2 | Comment 2 |
| ...             | ...               | ...       |
| Department n    | Survey Question n | Comment n |

If your data does not contain distinct questions or report grouping attributes, then do not include these columns in your dataset. The pipeline will then perform an attribute-agnostic analysis on the comments alone. Currently, the pipeline only supports analysis of English data.

2. Run the main analysis pipeline on your dataset by passing the filepath as a command argument.

<pre>$ python -m pnlp --path yourfilepath</pre>

This command will run a default analysis pipeline, outputting several log files and summary analytics including visualization of identified document clusters and sentiment labels. 

***

### Optional Arguments

You may pass several optional arguments to override the default pipeline configuration. To view these arguments please consult the output of the following command.

<pre>$ python -m pnlp --help</pre>

***


