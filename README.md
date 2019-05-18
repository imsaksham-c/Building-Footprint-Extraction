*This repository was originally at https://github.com/aiforearth/SpaceNetExploration*. For other Microsoft AI for Earth repositories, search for the topic `#aiforearth` on GitHub or visit them [here](https://github.com/search?l=&q=user%3Amicrosoft+topic%3Aaiforearth&type=Repositories).

# Building Footprint Extraction

## Overview
This repository contains a walkthrough demonstrating how to perform semantic segmentation using convolutional neural networks (CNNs) on satellite images to extract the footprints of buildings. We show how to carry out the procedure on an Azure Deep Learning Virtual Machine (DLVM), which are GPU-enabled and have all major frameworks pre-installed so you can start model training straight-away. We use a subset of the data and labels from the [SpaceNet Challenge](http://explore.digitalglobe.com/spacenet), an online repository of freely available satellite imagery released to encourage the application of machine learning to geospatial data.


## Data

### SpaceNet Building Footprint Extraction Dataset
The code in this repository was developed for training a semantic segmentation model (currently two variants of the U-Net are implemented) on the Vegas set of the SpaceNet building footprint extraction [data](https://spacenetchallenge.github.io/). This makes the sample code clearer, but it can be easily extended to take in training data from the four other locations.

The organizers release a portion of this data as training data and the rest are held out for the purpose of the competitions they hold. For the experiments discussed here, we split the official training set 70:15:15 into our own training, validation and test sets. These are 39 GB in size as raw images in TIFF format with labels.


### Generate Input from Raw Data
Instruction for downloading the SpaceNet data can be found on their [website](https://spacenetchallenge.github.io/). The authors provide a set of utilities to convert the raw images to a format that semantic segmentation models can take as input. The utilities are in this [repo](https://github.com/SpaceNetChallenge/utilities). Most of the functionalities you will need are in the `python` folder. Please read their instructions on the repo's [README](https://github.com/SpaceNetChallenge/utilities) to understand all the tools and parameters available. After using `python/createDataSpaceNet.py` from the utilities repo to process the raw data, the input image and its label look like the following:

![alt text](https://github.com/yangsiyu007/SpaceNetExploration/blob/master/visuals/sample_input_pair.png)




## Related Materials

Bing team's [announcement](https://blogs.bing.com/maps/2018-06/microsoft-releases-125-million-building-footprints-in-the-us-as-open-data) that they released a large quantity of building footprints in the US in support of the Open Street Map community, and [article](https://github.com/Microsoft/USBuildingFootprints) briefly describing their method of extracting them.

[Blog post](http://jeffwen.com/2018/02/23/road_extraction) and code on road extraction from satellite images by Jeff Wen on a different dataset.

SpaceNet [road extraction](https://spacenetchallenge.github.io/Competitions/Competition3.html) challenge.

[Tutorial](https://github.com/Azure/pixel_level_land_classification) on pixel-level land cover classification using semantic segmentation in CNTK on Azure.


