# Self Driving Car Model

## Project Description
This course aims to harness the power behind deep learning & computer vision to build a fully functional self-driving car. Cars are typically driven around and trained on real roads by manual drivers and then they are trained on data and clone the behavior of manual drivers!

I have used the Udacity simulator to take generate the training data images in the movement of the car (3 laps in both directions to avoid biased samples)
X-train = snapshot
y-train = steering angle
Model: We trained a CNN neural networks based on (X-train, y-train) data to adjust the steering angle. The next step was to use the trained model on top of the simulator. The task was to predict the correct angle based on the frame given (regression).

The trained model is then then used to drive the car anonymously of a different track.
