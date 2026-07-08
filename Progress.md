# Corn Disease Detection

## 2020

### October
* The dataset was made live on [Kaggle](https://www.kaggle.com/qramkrishna/corn-leaf-infection-dataset).
* I published a blog about data preparation on [q-viper.github.io](https://q-viper.github.io/2020/10/19/corn-infection-detection-data-preparation/).

### September
* I got help with internet access from Vikash Krishna and tried to test YOLO for corn on Colab.
* I used TrainYourOwnYolo for training. All of my images had to be rotated by 90 degrees for the annotations to work.
* Training was unsuccessful.

### August
* I wrote code to classify whether a patch of corn was infected or not. I had to create a custom data generator and feed it to the classifier.
* I wrote code to predict possible bounding boxes on images by training it on images and bounding boxes. I think it did not work.
* Both pieces of code were tested on local hardware, and the dataset was still not on the cloud.

### July
* As of July 26, nothing exciting was happening, but I was studying Region Proposal Networks. It was hard to understand other people's code, so I was thinking about writing my own version of an RPN.
* On July 22, I considered selling the infected/normal corn leaf image classification work to some MSc students.
* I needed to go to a Wi-Fi zone and stay there for days to train a detection model using GPUs.
* On July 11, I still had electricity problems and heavy rain.
* On July 11, I completed the annotation. I was still wondering whether I should do augmentation.
* Learned about some data generators for Keras.
* I wrote code to find image regions and also used some image segmentation code with the `selective-search` algorithm.
* On July 3, I wrote code to find annotations for augmented images.
* I found a bug with image reading and annotation in VoTT. If the image was rotated 90 degrees using `np.rot90`, the annotation became correct; otherwise, the annotation was wrong.
* That corn field was completely ruined.
* I took 2,009 images of freshly infected plants. There were more than 5,000 infected plants, but my phone battery was low.

### June
* Our corn was infected by insects and pests.
* I took the first photo of a corn plant on June 12.
* I was late to collect photos. Our field had only a few infected plants, and the other farmers' corn had already grown, so I was able to collect only 223 images.
* We also had no electricity for days, along with many weather changes.
* I wrote augmentation code to create 10 images from one image:
  * Flip x
  * Flip y
  * Flip xy
  * Brightness
  * Erosion
  * Dilation followed by erosion
  * Dilation
  * Erosion followed by dilation
  * Scale 0.75
  * Scale 0.5

### May
* Still no idea what to do next.

### April
* Still no idea what to do next.

### March
* I got an idea for a detection project involving hens.
* The idea was to detect dead hens, count them, and alert the owner. The plan was to collect an image dataset and later use a YOLO-like model.
* The project failed because data collection was not successful.

## 2018-2019
* I wrote some ML code, but I skipped the mathematics, so I still had problems understanding parts of it.

## 2017
* The idea originated in 2017. I wanted to work on an AI project in agriculture, but I was still a novice then.
