---
layout: post
title: Summaries of Machine Learning Resources
---

I thought it a good idea to have a place to reference all the material I have found, read and studied about Machine Learning, Deep Learning, Reinforcement Learning and Artificial Intelligence in general. I used to forget the deeper details and found myself re-reading articles, so I decided to make this post to easily find references in the future.

So without further ado, here are my summaries:

# Table of Contents

<!-- MarkdownTOC autolink="true" bracket="round" depth="0" style="unordered" indent="  " autoanchor="false" -->

- [Introductions](#introductions)
    - [What is holding you back from your machine learning goals?](#what-is-holding-you-back-from-your-machine-learning-goals)
- [Deep Learning](#deep-learning)
  - [Convolutional Neural Networks](#convolutional-neural-networks)
    - [A Beginner's Guide To Understanding Convolutional Neural Networks](#a-beginners-guide-to-understanding-convolutional-neural-networks)
    - [Video: Deep Visualization Toolbox](#video-deep-visualization-toolbox)
    - [A Beginner's Guide To Understanding Convolutional Neural Networks Part 2](#a-beginners-guide-to-understanding-convolutional-neural-networks-part-2)
    - [The 9 Deep Learning Papers You Need To Know About \(Understanding CNNs Part 3\)](#the-9-deep-learning-papers-you-need-to-know-about-understanding-cnns-part-3)
- [Reinforcement Learning](#reinforcement-learning)
    - [Youtube Playlist: Reinforcement Learning by Jacob Schrum](#youtube-playlist-reinforcement-learning-by-jacob-schrum)
    - [Youtube Playlist: Reinforcement Learning by David Silver](#youtube-playlist-reinforcement-learning-by-david-silver)
    - [Source Code: UC Berkeley CS188 Intro to AI - Course Materials](#source-code-uc-berkeley-cs188-intro-to-ai---course-materials)
  - [Videos about DeepMind](#videos-about-deepmind)
    - [The Future Of Artificial Intelligence by Demis Hassabis - DeepMind Founder](#the-future-of-artificial-intelligence-by-demis-hassabis---deepmind-founder)

<!-- /MarkdownTOC -->

# Introductions

#### [What is holding you back from your machine learning goals?](http://machinelearningmastery.com/what-is-holding-you-back-from-your-machine-learning-goals/)

It is a short motivational post with the goal to get you immediately started with Machine Learning. It tells you why you should start even if you don't feel ready, and urges you to start with small projects, with links on how to do so. 

I find it a nice post for starters, since when I began my quest in machine learning I took lot of courses and read lots of resources, but found myself not using or even needing more than half of it when I finally started working on real problems.

# Deep Learning

## Convolutional Neural Networks

#### [A Beginner's Guide To Understanding Convolutional Neural Networks](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)

This is the first post of a series of three that introduces Convolutional Neural Networks (CNNs) in order to get a deeper intuition into how they work. I had learned CNNs existed and that they were used for computer vision and automatic feature extraction, but until I found this post I hadn't grasped how exactly did it do that, and how and why was it different from normal neural networks or why did they need to be deep?

It doesn't touch everything, but the post is still great. I gives a clear example of how it find patterns in the first layers and how the connections are different from the first neural networks. It explains fairly well the convolutional layers, but leaves for later the ReLU and Pool layers. Here I found a link to a [video](https://www.youtube.com/watch?v=AgkfIQ4IGaM) which really gives you a greater vision of what convolution does and why it works great for computer vision.

I goes a little into the final fully connected layer, backpropagation, testing and touches a little of its nuances. It is not so superficial, but not as comprehensive as the convolution part (which is what the post is supposed to be about), and I think I understood what he wrote in those parts because I already knew about them.

A very good post overall, and fulfills its goal of introducing CNNs to a technical audience who wishes to understand how they work.

#### [Video: Deep Visualization Toolbox](https://www.youtube.com/watch?v=AgkfIQ4IGaM)

This is the video mentioned [above](#a-beginners-guide-to-understanding-convolutional-neural-networks). It is the best video I have found that literally shows in real-time how the neurons in a CNN activate with certain features, and how the deeper the layer, the broader part of the image that each neuron recognizes.

The first time I saw it was eye-opening, but after working more on convolution layers, I watched it again and found it to be more impressive than what I thought at first. Really recommended video to watch.

#### [A Beginner's Guide To Understanding Convolutional Neural Networks Part 2](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/)

This is the second post of the series stated [above](#a-beginners-guide-to-understanding-convolutional-neural-networks). Once again, still introductory but very well explained. Helped me understand stride and padding. Also briefly mentions the other types of layers in a CNN (ReLU and Pooling). Has a short paragraph on Dropout and Network in Network Layers, without going into details but linking to related papers. It ends with more links to research papers on object detection and segmentation.

#### [The 9 Deep Learning Papers You Need To Know About (Understanding CNNs Part 3)](https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)

Final post of the series following the one [above](#a-beginners-guide-to-understanding-convolutional-neural-networks-part-2). The title says it all, it summarizes 9 important papers. They are well explained for being short summaries, and have links to the papers. Very important resource to start learning more about the field. It basically shows an evolution of using deep learning for computer vision, the one I found the most fascinating was about Generative Adversarial Networks.

# Reinforcement Learning

#### [Youtube Playlist: Reinforcement Learning by Jacob Schrum](https://www.youtube.com/playlist?list=PLWi7UcbOD_0u1eUjmF59XW2TGHWdkHjnS)

After finding out that one of the technologies DeepMind used for their Atari playing implementation was Reinforcment Learning, I started looking for resources to understand what it was, and also how and why it worked. I found a lot of resources, none of which gave you such a good grasp as this video playlist. It is really introductory, but extremely clear to understand, and the most important points are properly covered.

#### [Youtube Playlist: Reinforcement Learning by David Silver](https://www.youtube.com/playlist?list=PL7-jPKtc4r78-wCZcQn5IqyuWhBZ8fOxT)

Weeks after watching these videos I came to find out that David Silver works for Google DeepMind, and according to his [public LinkedIn Profile](https://www.linkedin.com/in/david-silver) he is as of this writing working in self-driving cars, which is nonetheless totally awesome. 

That said, I haven't properly gone through his whole class. I have only seen the first 2 and last 3 videos. I found them a little boring like any other class that involves theory, so I just skipped through most of it. I know because of comments in many places, a lot of people idolize his class, and as stated in the first paragraph, I think the work he's done is great, but personally I didn't like his videos much. 

Though, it might be that I am too new to the subject to properly appreciate its contents, which no doubt must have its significance. However, just like I learned the first real-world practical application of linear algebra 3 years after graduating from my bachelors, I'd much rather learn by doing, and a good start is the code below.

#### [Source Code: UC Berkeley CS188 Intro to AI - Course Materials](http://ai.berkeley.edu/reinforcement.html)

I found this code which I believe is the starter code for [this class](https://www.youtube.com/channel/UCB4_W1V-KfwpTLxH9jG1_iA/videos). Lectures [10](https://www.youtube.com/watch?v=IXuHxkpO5E8&t) and [11](https://www.youtube.com/watch?v=yNeSFbE1jdY&t) are specifically about RL. The videos, although long, are quite helpful, and playing with the code is a great learning experience. I had to make some strange changes to play around with a dynamic epsilon, but in the end was possible. 

## Videos about DeepMind

#### [The Future Of Artificial Intelligence by Demis Hassabis - DeepMind Founder](https://www.youtube.com/watch?v=e0NuW1j9RPA&feature=youtu.be)

I have found quite a few videos of Demis Hassabis explaining about their game playing RL algorithms. I found this one in particular quite interesting, as it goes into a little more detail about some "unthinkable" things AlphaGo did when defeating Lee Sedol, like putting a piece in the fifth row instead of the fourth or third like traditional play, how Lee Sedol got up from the game, the commentary in English that thought it was a mistake, and that ultimately AlphaGo won that match in part because of that move. It was also the first place I saw him mentioning that AlphaGo had in some way showed intuition and creativity, attributes normally given to humans. 

Near the end around the 41 minute mark, it shows a graphic of how this same algorithm when put in charge of Google's data center greatly optimizes the power consumption, seemingly by around 40%. A little later, shows their results of WaveNet on speech quality and piano music. Overall, a very interesting video.