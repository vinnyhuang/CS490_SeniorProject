# CS 490 - Senior Project

Title: Implementation of a Convolutional Nural Network and Visualization Tool for Learning of Steering Angles in Self-Driving Cars

Abstract:
As the self-driving car industry grows and vehicle autonomy plays a larger and larger role in mainstream cars, safety becomes an increasingly important issue.  The FLINT group at Yale has established Y Driving, a self-driving car lab with the purpose of researching new methods in autonomous vehicular safety, navigation, and deployment.  This semester, I explored solutions for learning steering angles that could be used for the lab’s model cars, and I implemented a convolutional neural network (CNN) based on the NVIDIA team’s seminal paper.  Neural nets are commonly critiqued for their black boxes opacity, and a major area of machine learning today seeks to increase their transparency.  With safety such a critical issue, it is important to maintain a level of transparency in neural nets involved in autonomous vehicles.  To accomplish this, I built out an implementation of VisualBackProp, a visualization technique also designed by the NVIDIA team, used to visualize which pixels and features in an input image contribute most to the predictions made by a CNN.  I adapted my CNN and VisualBackProp implementations to run independently on Udacity’s self-driving car simulator and on real world street data.  The systems are generalized such that any data set can be used to train the CNN model used to generate predictions on steering angles, after which the VisualBackProp program can be used to identify the key activation pixels in any input image for the neural net.  As such this solution can be easily ported to work on driving data collected from Y Driving’s model cars to autonomously steer the vehicles and generate visual representations of the net’s decision-making processes.

## Environment Setup

To begin, set up the required environments:

```
conda env create -f environment.yml
conda env create -f environment-train.yml
```

The only difference between these environments is the version of Keras installed.  Because the training code (model.py) is optimized for Keras 1, running it using 490-train allows it to run much faster.

Before any training (model.py), run the command:
```
source activate 490-train
```

Before running any other of the programs (drive.py, vbp_display.py), run the command:
```
source activate 490-run
```

On a zoo machine, you may have to run instead:
```
source /usr/local/anaconda3/bin/activate 490-run
source /usr/local/anaconda3/bin/activate 490-train
```

### Udacity Simulator
First, download and install the simulator from [Udacity's github](https://github.com/udacity/self-driving-car-sim).

Collect driving data in training mode.  The process will produce a data folder containing a subfolder titled IMG and a ```driving_log.csv``` file.

To train, run (in vincent490/simulator):
```
python model.py -d `[data folder, the parent of IMG]` -o 'false'
```
The -o 'true' flag only saves models at checkpoints where the model performs better by Keras's metrics.  I have found this to be undesirable.

```model.py``` will produce multiple .h5 files that are trained networks that can be used for steering control.  To test it, open the simulator and enter autonomous mode.  Then run the following command, which takes a mandatory parameter (.h5 file path):
```
python drive.py `[.h5 file]`
```
If you don't wish to train your own model, you can use the one I trained, ```titled model_sim.h5```.


Once you have a suitable trained net, run VisualBackProp.  Then following command takes two mandatory parameters (a .h5 file path and an image file path):
```
python vbp_display.py `[.h5 file]` `[img file]`
```
I have also provided a sampel image that can be used alongside my ```model_sim.h5```.  Run:
```
python vbp_display.py model_sim.h5 center_2018_04_25_17_33_36_293.jpg
```

### Street Data Runner
Download the data from this [link](https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view?usp=sharing).  Unzip the file and export the file driving_dataset/.  This repo contains a vincent490/street-driving/data.csv file.  Copy that into the driving_dataset/ folder.  Then run ```model.py```, but make sure that the -d parameter is the path of the folder that contains the both the images and the .csv file (unlike last time, when it was just the .csv file at that level).  An example as such:
```
python model.py -d driving_dataset -o 'false'
```

To see the system "drive" the car, run 
