# LearningML

This is my attempt to (finally!) do some hands-on learning about machine learning.
I've learned a bit about the basic theory of ML but I've never tried it myself.

This repo will be mostly copy-pasted code from the TensorFlow Keras [tutorials](https://www.tensorflow.org/tutorials/keras/classification).

## Notes
To force myself to do some thinking, I'll take notes on what the code is doing.
Hopefully this will prevent me from simply blindly copy-pasting.

Theoretically as I continue going through the tutorials I should be able to figure out some of my questions and remove them.

### Things I think I understand
* The clothing classifier is a supervised learning model because we're giving it datasets that are already labeled for training.
* This classifier gives a probability for each possible label, and we pick the highest probability as its prediction.
* A machine learning model is created by connecting several simple layers. Some layers perform simple data manipulation (like `Flatten`) and others are trained (like `Dense`, whatever that means).
* "Logits" are (is?) a vector represent the raw predictions generated by a classifier, which are usually then normalized so that humans can understand them.
* `Sequential` takes in multiple layers and executes them in order.
* The classifier can be very confident and still very wrong.

### Things I don't understand
* Why do we need to change the image pixel values to be floats from 0 to 1?
  * I tried commenting out those lines and the model became less accurate.
* Does the `Dense` layer imply the existence of a `Sparse` layer? If so, why are we using `Dense` and not `Sparse`?
* What is an `adam` optimizer?
* What does `relu` mean?
* What the heck is a `SparseCategorialCrossentropy`?
  * This sounds like a randomly generated name that Replit would give you.
* Why does do different runs yield such different predictions?
  * The 13<sup>th</sup> image (which is a sneaker but sometimes gets mixed up with a sandal) got a 98% chance of being a sandal on the run after it was classified as a sneaker with around 60% confidence.
