# Bread Classification App

## TL;DR
Visit https://huggingface.co/spaces/maianume/computer_vision_toast and play!

## Introduction
This repository contains a machine learning application built with Gradio and OpenCV for classifying images of bread based on their toastiness level. The app allows users to select an image of bread and then classifies it to find the most similar reference image from a predefined dataset.

## Installation

To run the application, you need to have Python installed on your system. This app has been tested with Python 3.8+. You also need to install the required dependencies.

First, clone this repository to your local machine:

```bash
git clone https://github.com/maianumerosky/computer_vision_toast
```

Then, install the required Python packages:
```
pip install -r requirements.txt
```

## Usage

To launch the application, run the following command in the terminal:

```
python gradio_breads_app.py
```

After the application starts, you will see a URL in the terminal that you can visit in your web browser to interact with the app.

Select an image of bread from the provided options, and click the "Classify Bread" button. The application will then display the image you selected and the most similar reference image based on the classification.
<<<<<<< HEAD

=======
>>>>>>> origin/master
## How It Works

The app uses OpenCV to process images and extract features, and then compares these features with those of reference images to find the most similar match. The comparison is based on the histograms of the images in the CIELAB color space.
