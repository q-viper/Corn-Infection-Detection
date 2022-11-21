# Corn Disease Detection

## 2017
* Idea origin on 2017(wished to do some AI project on Agriculture but i was novice back then).

## 2018- 2019
* Did some of ML codes but skipped mathematics(so i am still having problems).

## 2020
### March
* Got an idea to do some detection project on hens.
* The idea was to detect died hens and count them then warn the owner. The concept was to collect image dataset and use YOLO like model later.
* But failed because data collection was not succesful.

### April
* Still no idea what to do next.

### May
* Still no idea what to do next.

### June
* Our corn got infected due to some swamp insects and pests. 
* Then i took first photo of corn's plant on June 12.
* I was late to collect photos and our field has only few infected and other farmer's corn was already grown. So i was able to collect only 223 images.
* Also electricity was gone for days and many climate changes.
* I wrote some agumentation code to make 10 images from one image.
    * Flip x
    * Flip y
    * Flip xy
    * Brightness
    * Erosion
    * Dilation followed by Erosion
    * Dilation
    * Erosion followed by Dilation
    * Scale 0.75
    * Scale 0.5

### July
* Took 2009 images of freshly infected plants. There were more than 5k infected plants but my phone battery was low.
* Those field of corn was totally ruined.
* Found a bug of image read and annotation of VOTT. If we rotate image 90 degree then the annotation becomes true else annotation becomes wrong. (using np.rot90)
* Wrote a code on 3rd day to find annotation of agumented images. 
* Wrote some codes to find region of images and also used some image segmentation codes but used `selective-search` algorithm.
* Learned about some data generators for Keras.
* 11th day, completed annotation. Still having question should i do agumentation? 
* Having electricity problem still on 11th day and heavy rain. 
* Should go to wifi zone and stay for days to train a detection model using GPUs.
* July 22, about to sell classification of infected/normal corn leaf image to some MSc students.
* As for july 26, nothing exciting happening. But studying about Region Proposal Network. It is hard to understand code of other peoples so i am thinking about writing my own version of RPN.

### Agust
* Wrote a code to classify whether a patch of corn is infected or not. Had to make custom data generator and then feed it to classifier.
* Worte a code to predict possible Bounding Boxes on images by training it on images and bounding boxes. I think it doesn't work.
* Both above codes were tested on local hardware and still dataset is not on cloud.

### September
* Got help on my internet by Vikash Krishna and trying to test YOLO for corn on colab.
* Using TrainYourOwnYolo for training. My entire images has to be rotated by 90 and only the annotation works.
