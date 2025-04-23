<p align="center">
<h3 align="center">Module C6 Project</h3>

  <p align="center">
    Project for the Module C6-Video Analysis in Master's in Computer Vision in Barcelona.
<br>
    <a href="https://github.com/mcv-m6-video/mcv-c6-2025-team1/issues/new?template=bug.md">Report bug</a>
    ·
    <a href="https://github.com/mcv-m6-video/mcv-c6-2025-team1/issues/new?template=feature.md&labels=feature">Request feature</a>
  </p>
</p>


> [!IMPORTANT]
> The final presentation is available [here](https://docs.google.com/presentation/d/1Qbxnnqear_4xCycYvNFZWbpnvOAXVVJ_aQJnRVKDvvQ/edit?usp=sharing). If for some reason you don't have permissions to access it, contact any of the administrators of this repository.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Project Structure](#project-structure)
- [WEEK 1](#week-1)
- [WEEK 2](#week-2)
- [WEEK 3](#week-3)
- [WEEK 4](#week-4)
- [WEEK 5](#week-5)
- [WEEK 6](#week-6)
- [WEEK 7](#week-7)
- [Team Members](#team-members)
- [License](#license)

## Introduction

This project is developed as part of the Master's program in Computer Vision in Barcelona, specifically for the course **C6: Video Analysis** during the third academic semester. 

The goal of this project is to implement computer vision techniques for **road traffic monitoring**, enabling the detection and tracking of vehicles in video footage from multiple cameras. The system is designed to analyze traffic flow by applying the following key methodologies:  

- **Background modeling**: Establishing a model to differentiate between static background and moving objects.  
- **Foreground detection**: Identifying vehicles by segmenting them from the background.  
- **Motion estimation**: Using optical flow techniques to estimate vehicle movement.  
- **Multi-object tracking**: Combining detections and motion estimation to track multiple vehicles across video frames and camera viewpoints.  

This project aims to contribute to intelligent traffic monitoring systems, improving road safety, traffic management, and urban mobility analysis.  

## Installation

This section will guide you through the installation process of the project and its testing.

### Prerequisites
The following prerequisites must be followed:
- Python >= v3.12

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yeray142/mcv-c6-2025-team1
   cd mcv-c6-team1

   # To install all third party tools
   git submodule update --init --recursive
   ```

2. **Navigate to the corresponding week's folder:**
   For example, to enter the folder for week 1:
   ```bash
   cd week1
   ```

3. **Choose one of the following methods to set up your environment:**

#### Option A: Using Python Virtual Environment

1. **Create a virtual environment:**
   ```bash
   python -m venv env
   ```

2. **Activate the virtual environment:**
   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```
   - On MacOS/Linux:
     ```bash
     source env/bin/activate
     ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

#### Option B: Using Conda Environment (Recommended)

1. **Create a conda environment from the environment.yml file:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the conda environment:**
   ```bash
   conda activate mcv-c6-2025
   ```
   
## Project Structure

Within the downloaded repository, you'll find the following directories and files, logically grouping common assets. The data folders need to be downloaded and decompressed from the provided links:

- **AICity_data:** [Download here](https://e-aules.uab.cat/2024-25/pluginfile.php/577737/mod_page/content/98/AICity_data.zip)
- **results:** [Download here](https://e-aules.uab.cat/2024-25/pluginfile.php/577737/mod_page/content/98/results_opticalflow_kitti.zip)
- **ai_challenge_s03_c010-full_annotation.xml:** [Download here](https://e-aules.uab.cat/2024-25/pluginfile.php/577737/mod_page/content/98/ai_challenge_s03_c010-full_annotation.xml)

Once downloaded and extracted, the project structure will look like this:

    Team1/
    ├── data/
    │   ├── AICity_data/
    │   ├── results/
    │   └── ai_challenge_s03_c010-full_annotation.xml
    ├── week1/
    │   └── ...
    ├── week2/
    │   └── ...

<h2>WEEK 1</h2>

The contents of the first week are in the folder `week1`. The `README` file can be found in [here](week1/README.md).

<h2>WEEK 2</h2>

The contents of the second week are in the folder `week2`. The `README` file can be found in [here](week2/README.md).

<h2>WEEK 3</h2>

The contents of the third week are in the folder `week3`. The `README` file can be found in [here](week3/README.md).

<h2>WEEK 4</h2>

The contents of the fourth week are in the folder `week4`. The `README` file can be found in [here](week4/README.md).

<h2>WEEK 5</h2>

The contents of the fifth week are in the folder `week5`. The `README` file can be found in [here](week5/README.md).

<h2>WEEK 6</h2>

The contents of the sixth week are in the folder `week6`. The `README` file can be found in [here](week6/README.md).

<h2>WEEK 7</h2>

The contents of the seventh week are in the folder `week7`. The `README` file can be found in [here](week7/README.md).

## Team Members

This project was developed by the following team members:

- **[Judit Salavedra](https://github.com/juditsalavedra)**
- **[Judith Caldés](https://github.com/judithcaldes)**
- **[Carme Corbi](https://github.com/carmecorbi)**
- **[Yeray Cordero](https://github.com/yeray142)**

## License
The MIT License (MIT). Please see [LICENSE File](LICENSE) for more information.


