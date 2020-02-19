---
layout: page
title: Neural Networks (and Machine Learning) for the Non-Technical
---

##### Brains, Machines, and AI; Oh my!

![[[Image Source](https://blog.adva.com/en/is-disaggregation-an-evolutionary-necessity)]](https://cdn-images-1.medium.com/max/3200/0*8e6cFyGZ3IFFGGUs.)
<span class='caption'>*[[Image Source](https://blog.adva.com/en/is-disaggregation-an-evolutionary-necessity)]*</span>

I recently moved to a new city (Padres fans?!… Anyone??) and I’ve been struggling with how to answer the inevitable get-to-know-you question/conversation filler:
> # *New Friend: “So… what do you do?”*
> # *Me: Uhmm, er, well… I used to work in web development but I’m getting into machine learning now.*
> # *New Friend: “So that’s like artificial intelligence right?”*
> # *Me: \*crickets\**

I haven’t found a simple way to explain how I spend my days (when not playing volleyball). Perhaps I don’t understand it well enough myself. So let’s explore!

Disclaimer: The following is neither an exhaustive nor necessarily accurate survey of the field of Machine Learning. This ain’t wikipedia…¯\\\_(ツ)\_/¯

This post is simply a synthesis of my current understanding and why I think it’s a compelling area of study. Consume generously salted.

***

## What is Machine Learning

Machine learning offers a way to solve complicated problems by training an algorithm (program) on examples rather than explicitly coded instructions.

You might think, huh, that’s a weird way to go about it. Machines are pretty dumb after all. Wouldn’t it be easier to use our superior logic and reasoning to think about a problem in a systematic way and then somehow “program” a computer to do exactly what we want it to do?

Well… yea. That makes sense and works really well for lots of problems. Need help calculating forces when constructing skyscrapers, airplanes, and rocketships? Let’s engineer some software to help you with that! Need a tool which allows air traffic controllers to monitor and direct thousands of concurrent flights to avoid fiery, screaming deaths? Computer programs can help. Want to build a globally distributed information platform so we can share dank memes and cat videos? Programming!

What about recommending other videos I might like? Or translating a street sign in a foreign land to a language I understand? Or a little box that I talk to and it responds (semi) intelligently?
> # **crickets**

Turns out some tasks are not so easily programmable. Even for really smart software engineer wizards. It can be done but the results are pretty clunky.

![[[Image Source](https://www.pinterest.com/pin/502292164666616725)]](https://cdn-images-1.medium.com/max/2000/0*DdeRPebRH09GWxe9.)
<span class='caption'>*[[Image Source](https://www.pinterest.com/pin/502292164666616725)]*</span>

But why? Our brains handle stuff like this easily. I can instantly recognize my friend’s face in a picture on facebook even if she’s in a dimly lit bar wearing a ridiculous, fake mustache. I can (roughly) understand a person talking to me in English even with a funny accent. I can drive a car while listening to the radio, slurping my big gulp, and texting with friends! (Just kidding, I can’t. But teenagers can!)

These types of tasks all come intuitively to us and have a high degree of variability. It’s very hard to formulate things like this with explicit rules. Try describing how you recognize someone’s face or understand a sentence…

Language acquisition is a useful analog to machine learning. We are immersed in our language since birth. From interacting with our parents and siblings, to television programs, signs, toys, radio, etc. Language data is flowing into our brains from all around and as we grow we begin forming rudimentary word-image connections. Soon we start to guess the names of things we see in the world and get feedback on our guesses.
> # Mother: “Yes honey, that’s a fire truck.” or “No Johnny, that’s an Eastern gray squirrel.”

Our language becomes richer and more abstract as we continue to learn through interactions.

All of this happens in our brains before we even set foot in a school. Which is to say before we are taught explicit rules about grammar.
> # Johnny: “I talk English good.”
> # Teacher: “No Johnny, You talk English *well*.”

Want to learn a new language? Go abroad and immerse yourself in that culture. Interact with people. Get feedback. Repeat.

Want to train an algorithm to learn something? Immerse it in the type of data you want it to learn. Have it make guesses. Provide feedback. Repeat.

Say we want a machine learning algorithm to help us classify 5 different types of trees in our area. We begin by showing it a bunch of examples of each type randomly mixed together and grading it on how accurately it guessed the correct type. When it first starts out it’s randomly guessing and has a 1 in 5 chance of getting it right. Correct guesses result in some internal parameters being adjusted a little in one direction and increasing the ‘connection’ between that set of inputs and that guess. Incorrect guesses spin the internal knobbies in the other direction and decrease the likelihood of repeating that guess in response to those inputs. As training continues, the algorithm gets better and better at classifying images of trees.

Two important things to note here:

1. As you can probably surmise, the amount of quality data is the single most important component of effective machine learning. If we only have a few examples per class our algorithm will get really good at accurately guessing those examples but not very good at generalizing beyond. Accurate generalization (correctly classifying new images) is the goal. The more relevant data an algorithm learns from, the better it generalizes.

1. Say we now want to classify plants or animals in our area. The underlying architecture of our algorithm doesn’t need to change at all. Machine learning algorithms are (nearly) infinitely flexible learners. Like children, they learn what you teach ‘em.

***

Before moving forward, I want to clarify my use of the term “machine learning”.

The above example is an area of machine learning called **supervised learning**. Supervised learning generalizes *existing* patterns (i.e. labels) to *new* data.

Discovering *new* patterns in *existing* data is called **unsupervised learning**. A combination of the two represents a potential path towards what normal folks think of as artificial intelligence. I’ll touch on this shortly but that’s a topic for another post.

The realm of supervised learning is further divided into two areas: **regression** and **classification**. The tree example above is a classification problem. Inputs are classified, or categorized. Regression means predicting along a continuum, or spectrum, of values. E.g. sales forecasting, predicting temperature or stock price, etc.

Our supervised tree classifier above implicitly leverages a type of machine learning architecture called an **artificial neural network**. Neural networks (and from here on out I’m using this term to refer to artificial neural networks) power speech recognition systems, image classification, machine translation, self-driving cars, recommendation engines, facial recognition, language processing, and lots of other cool stuff happening in the field today. Neural networks are what we’ll be learning more about in the rest of this post. But first, SkyNet…

![I’ll be back… to explain neural networks in just a sec! [[Image Source](http://www.blastr.com)]](https://cdn-images-1.medium.com/max/2000/0*qa_id9VFgmSHvskI.)
<span class='caption'>*I’ll be back… to explain neural networks in just a sec! [[Image Source](http://www.blastr.com)]*</span>

***

## What it isn’t

As shown above, machine learning (and neural networks) approximate human learning. This is fundamentally different than approximating human thinking, or **artificial intelligence**.

AI is a loaded term and is best divided into two different types.

Artificial general intelligence (AGI), or strong AI, is where a machine can think as well as a human. Humans have remarkable cognitive flexibility. We dynamically adapt to changing situations and can translate concepts between disparate fields with ease, or at the very least, without catastrophic system failure. State of the art machine learning isn’t even close to this point. This is the realm of science fiction. Terminator lives here.

Narrow artificial intelligence, or weak AI, leverages machine learning to accomplish a specific task. Neural networks and all the rad stuff mentioned above fall into this category.

The reason I bring up the distinction between the two types is because when I hear the term artificial intelligence I think about strong AI (AGI) and I suspect others do as well. Machine learning is not strong AI (AGI) and it’s [debatable](https://towardsdatascience.com/no-you-cant-get-from-narrow-ai-to-agi-eedc70e36e50) whether it even lies on the path towards AGI.

If you want an absolutely fascinating long read about artificial intelligence see this Wait But Why [post](https://waitbutwhy.com/2015/01/artificial-intelligence-revolution-1.html). (and continue reading because Tim Urban’s writing is fantastic…)

***

## How Neural Networks work

Neural Networks (NN) are comprised of layers of interconnected nodes, or neurons. Each neuron has a bias value. Connections between neurons have a weight value. Weights and biases are initially set randomly (or in a normal distribution) and it is the tuning of these parameters which approximates learning. Intermediate layer neurons have an activation threshold which determines if their data flows forward to the next layer.

![](https://cdn-images-1.medium.com/max/3200/0*vwrFxtKnKo6wWN1V.)

The image above represents our example tree classifier as a single hidden-layer neural network.
> **Note**: This section contains general explanations as well as some slightly more technical details formatted like this. Feel free to skip the technical stuff.

A layer can be thought of as a chain of simple mathematical equations. As an input value travels along a synapse towards a neuron it is transformed by a linear function (i.e. multiplying the input by the weight). All of the linear transforms moving into a neuron are summed together with the neuron bias, and if that value is above a certain threshold it gets passed forward.
> The activation function for our hidden layer is called a [Rectified Linear Unit](http://cs231n.github.io/neural-networks-1/#actfun) (Relu). It’s a fancy term for a threshold of zero. Values above zero flow forward.

A network can have many of these hidden layers stacked between the input and output making it a **deep neural network**. Our toy example is shallow, with only one hidden layer.

The output layer operates in the same way as a hidden layer except that the number of neurons is determined by the number of classification categories and the activation function yields a prediction probability for each category.
> In our case we might use [Softmax](https://en.wikipedia.org/wiki/Softmax_function): the exponent of each output divided by the sum of all exponents.

Ok, so we show our neural network a few trees and as we expect it performs pretty poorly. Now what? We need a way to quantify how wrong our predictions are generally and then try to minimize that number. Enter the **cost function**, or loss function. There are many to choose from and the choice depends on the type of problem you are trying to solve. Choosing the right loss function for the problem domain is critical but the specifics of how the loss is calculated is not important to us right now.
> [Categorical cross entropy](https://en.wikipedia.org/wiki/Loss_functions_for_classification) is a common choice for classification and is one measure of how much our predicted values differ from the actual values.

We now have a loss value after running our data forward through the network. This value can be thought of as an altitude in the loss landscape.

![[[Image Source](http://www.adalta.it/Pages/-GoldenSoftware-Surfer-010.asp)]](https://cdn-images-1.medium.com/max/2000/0*IfCUUpyolchHTLBq.)
<span class='caption'>*[[Image Source](http://www.adalta.it/Pages/-GoldenSoftware-Surfer-010.asp)]*</span>

The loss landscape above represents a loss value (z axis or height/color) with respect to two parameters (mapped on the x and y axes). In reality our neural network has more than just two parameters and thus more dimensions than can be visualized but it helps to think about it in our familiar three dimensional space.

After our first run through the network, we’re probably high up on that peak in the right hand corner and we want to move downhill towards the happy sea of low loss and good generalization:) But we don’t yet know what our loss landscape looks like. We only know our current altitude so far. To determine our landscape we use a technique called **backpropagation**. This calculates the effect each parameter has on our overall loss and gives us a topographical map of our loss landscape, or gradient.
> Applying the [chain rule](http://colah.github.io/posts/2015-08-Backprop/) backwards through each layer yields the partial derivative of our loss function with respect to each parameter.

Using this gradient map we know which direction to update each of the parameters in our network in order to take a step downhill and lower our loss. The size of the step we take is called the **learning rate** and the process of iteratively optimizing our loss function by updating parameters via backpropagation is called **gradient descent**.

Training a neural network simply means running input values forward through our network to calculate a loss and computing gradients backwards to make small adjustments to the parameters. As the loss decreases with more training, a neural network ‘learns’ to distinguish relevant features for a specific set of data.

Important things not mentioned:

* [Input normalization](http://cs231n.github.io/neural-networks-2/#datapre) — Before we begin training our neural network it helps to normalize our input data (e.g. scaling all of our input values to fall within the same range). This allows the network to train faster and generalize better.

* CNNs for images — For image classification problems like the one above, a better approach would be to use a [Convolutional Neural Network](http://cs231n.github.io/convolutional-networks/). This is a topic for another post but CNNs explicitly maintain a spatial structure which makes sense for image processing where spatial context matters.

* [Overfitting](http://wiki.fast.ai/index.php/Deep_Learning_Glossary#Overfitting) — Overfitting occurs when we train a neural network for too long on a limited set of input data. The network gets better and better at predicting training data but worse at generalizing to unseen data.

* [Regularization](http://cs231n.github.io/neural-networks-2/#reg) — Methods to prevent overfitting in neural networks. (e.g. L2, dropout, weight decay)

* [Stochastic gradient descent ](http://wiki.fast.ai/index.php/Gradient_Descent#Stochastic_Gradient_Descent)— In practice, we often have more data than we can fit into memory. SGD breaks data into random batches and runs through our process above one batch at a time.

***

I am heavily indebted to the following phenomenal sources for much of this information. If you would like to learn more about neural networks and things I’ve mentioned above check out stuff from these fine folks:

1. Jeremy Howard’s [fastai](http://www.fast.ai/) course. Hands down the best introduction to neural networks. Perfect for folks who want to build cool stuff without getting bogged down in the weeds of understanding exactly how everything works first.

1. Adam Geitgey’s [Machine Learning is Fun!](https://www.machinelearningisfun.com/) website. Step by step examples and great explanations on really cool projects using machine learning (like generating super mario levels and facial recognition).

1. Michael Nielsen’s [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) online book. Covers neural networks in depth and explains the mathematical underpinnings of it all. Of particular note is a cool interactive demo of why NNs function as universal approximators.

1. Stanford University’s [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/) course. An alternative to Jeremy Howard’s top down approach. This course is very much concerned with the mathematical weeds and teaches CNNs from the bottom up.

1. Other super smart people with a knack for simple explanations:

* [Chris Olah](http://colah.github.io/)

* [Andrej Karpathy](https://medium.com/@karpathy/)
