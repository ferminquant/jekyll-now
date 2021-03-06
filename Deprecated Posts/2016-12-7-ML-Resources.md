---
layout: post
title: Summaries of Machine Learning Resources
---

I thought it a good idea to have a place to reference all the material I have found, read and studied about Machine Learning, Deep Learning, Reinforcement Learning and Artificial Intelligence in general. I used to forget the deeper details and found myself re-reading articles, so I decided to make this post to easily find references in the future.

At first, I started with the ones I remembered had been the most useful. Then I started adding every one I read as per my original purpose, but it didn't work out, because I found myself not completely reading them but skimming through most. Also, not all were really useful for the future, so I removed the ones without any relevant content for my future reference.

So without further ado, here are my summaries:

# Table of Contents

<!-- MarkdownTOC autolink="true" bracket="round" depth="0" style="unordered" indent="  " autoanchor="false" -->

- [Start with Machine Learning](#start-with-machine-learning)
    - [Why becoming a data scientist might be easier than you think](#why-becoming-a-data-scientist-might-be-easier-than-you-think)
    - [Why Get Into Machine Learning?](#why-get-into-machine-learning)
- [Data Science Techniques](#data-science-techniques)
    - [Forget your fancy data science, try overkill analytics](#forget-your-fancy-data-science-try-overkill-analytics)
- [Deep Learning](#deep-learning)
    - [Youtube Playlist: Deep Learning SIMPLIFIED](#youtube-playlist-deep-learning-simplified)
    - [My Top 9 Favorite Python Deep Learning Libraries](#my-top-9-favorite-python-deep-learning-libraries)
  - [Convolutional Neural Networks](#convolutional-neural-networks)
    - [A Beginner's Guide To Understanding Convolutional Neural Networks](#a-beginners-guide-to-understanding-convolutional-neural-networks)
    - [Video: Deep Visualization Toolbox](#video-deep-visualization-toolbox)
    - [A Beginner's Guide To Understanding Convolutional Neural Networks Part 2](#a-beginners-guide-to-understanding-convolutional-neural-networks-part-2)
    - [The 9 Deep Learning Papers You Need To Know About \(Understanding CNNs Part 3\)](#the-9-deep-learning-papers-you-need-to-know-about-understanding-cnns-part-3)
- [Reinforcement Learning](#reinforcement-learning)
    - [Youtube Playlist: Reinforcement Learning by Jacob Schrum](#youtube-playlist-reinforcement-learning-by-jacob-schrum)
    - [Youtube Playlist: Reinforcement Learning by David Silver](#youtube-playlist-reinforcement-learning-by-david-silver)
    - [Source Code: UC Berkeley CS188 Intro to AI - Course Materials](#source-code-uc-berkeley-cs188-intro-to-ai---course-materials)
  - [DeepMind](#deepmind)
    - [Video: The Future Of Artificial Intelligence by Demis Hassabis - DeepMind Founder](#video-the-future-of-artificial-intelligence-by-demis-hassabis---deepmind-founder)
    - [Video: Demis Hassabis, CEO, DeepMind Technologies - The Theory of Everything](#video-demis-hassabis-ceo-deepmind-technologies---the-theory-of-everything)
    - [Video: Demis Hassabis, Google DeepMind - Artificial Intelligence and the Future](#video-demis-hassabis-google-deepmind---artificial-intelligence-and-the-future)
    - [Video: Inside DeepMind](#video-inside-deepmind)

<!-- /MarkdownTOC -->

# Start with Machine Learning

#### [Why becoming a data scientist might be easier than you think](https://gigaom.com/2012/10/14/why-becoming-a-data-scientist-might-be-easier-than-you-think/)

An interesting read, where it explains how new data scientists who have been successful at Kaggle competitions have quickly risen without lengthy and formal education. 

#### [Why Get Into Machine Learning?](http://machinelearningmastery.com/why-get-into-machine-learning/)

It has a strange matrix at the end, which I did not understand on its own before reading the whole post. However, after reading the post it makes sense. I identify more with the practitioner row of the matrix, and it gives suggested resources in which to begin your learning of machine learning.

# Data Science Techniques

#### [Forget your fancy data science, try overkill analytics](https://gigaom.com/2012/09/21/forget-your-fancy-data-science-try-overkill-analytics/)

A fast read about how a Kaggle competition winner used simple algorithms to win, an average of generalized linear regression and random forest. He uses the term overkill analytics as using computing power as the main source from which to get great results, instead of complicating the problem with a bunch of overengineering. His philosophy makes some sense, but in [his explanation of how he won](http://www.overkillanalytics.net/kaggles-wordpress-challenge-the-like-graph/) he admits to doing some previous work to make it computationally feasible. It is interesting to know he could win like that, it reminds me of [ensemble learning](https://en.wikipedia.org/wiki/Ensemble_learning), where multiple predictions are combined to get a better one.

# Deep Learning

#### [Youtube Playlist: Deep Learning SIMPLIFIED](https://www.youtube.com/playlist?list=PLjJh1vlSEYgvGod9wWiydumYl8hOXixNu)

An excellent series of videos explaining the concepts of deep learning at a high level. It does not go much into details, which is why it is great. It works more on the intuition, to get you to understand what deep learning does. Here I found out about the different algorithms, the differences between normal neural networks, different libraries and their features. It is really worth in general, but even more so when you are beginning in deep learning.

#### [My Top 9 Favorite Python Deep Learning Libraries](http://www.pyimagesearch.com/2016/06/27/my-top-9-favorite-python-deep-learning-libraries/)

Seems to be a useful blog post about deep learning tools recommended for use. He recommends Keras the most, but writes a small summary of quite a few.

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

## DeepMind

#### [Video: The Future Of Artificial Intelligence by Demis Hassabis - DeepMind Founder](https://www.youtube.com/watch?v=e0NuW1j9RPA&feature=youtu.be)

I have found quite a few videos of Demis Hassabis explaining about their game playing RL algorithms. I found this one in particular quite interesting, as it goes into a little more detail about some "unthinkable" things AlphaGo did when defeating Lee Sedol, like putting a piece in the fifth row instead of the fourth or third like traditional play, how Lee Sedol got up from the game, the commentary in English that thought it was a mistake, and that ultimately AlphaGo won that match in part because of that move. It was also the first place I saw him mentioning that AlphaGo had in some way showed intuition and creativity, attributes normally given to humans. 

Near the end around the 41 minute mark, it shows a graphic of how this same algorithm when put in charge of Google's data center greatly optimizes the power consumption, seemingly by around 40%. A little later, shows their results of WaveNet on speech quality and piano music. Overall, a very interesting video.

#### [Video: Demis Hassabis, CEO, DeepMind Technologies - The Theory of Everything](https://www.youtube.com/watch?v=rbsqaJwpu6A)

A short video by Demis Hassabis, where he explains a little about his background and how he got to where he is with DeepMind. He states he is getting inspired by neuroscience to solve AI, as a way to truly understand our brains. 

The DeepMind mission is 1. Solve Intelligence and 2. Use it to solve everything else

Later he gives a quick overview of how they made the Atari gaming, with examples of a few games. His dream is to make possible AI scientists or AI-assisted science. He concludes that the conference is called the theory of everything, because he believes that in order to find it, we might need to solve intelligence first.

#### [Video: Demis Hassabis, Google DeepMind - Artificial Intelligence and the Future](https://www.youtube.com/watch?v=f71RwCksAmI)

I think this is by far the most interesting video about AlphaGo. It is a conference in Korea, and it seems to have been in the middle of the matches with Lee Sedol, since at 40:47 he states that he wishes Lee Sedol the best luck in the next 3 games (they played 5 in total). He states that AlphaGo is around 10% of what they do in DeepMind. During the first half it is basically the same presentation of his other videos, a summary of the Atari games, but shows a little more games. He starts with AlphaGo at around 21:50.

At 28:20 he starts explaining about the two neural networks they trained for AlphaGo. The first one, called the policy network, used data from online matches through supervised learning to mimic amateur players. Then used reinforcement learning to make it play against itself, optimizing for win rate, until creating a new policy network that could beat the first one 80% of the time. Then grabbed that network and made it play itself 30 million times.

With the data generated above (30 million games with their outcomes), they made an evaluation function, the second neural network that was trained to predict the outcome given a game state, this was called the value network. Its purpose was to get an idea of who was winning with the game in progress.

To summarize both neural networks, the policy network predicts a probability for the next best move(s), and the value network is a range to indicate who is winning given a current board state. At around 31:45 he begins explaining the evaluation mechanism to choose the next move using both networks. They get the top 3 or 4 most probable moves (using the policy network) and perform a tree search expanding it. While it expands it uses the value network and a rollout policy (he explains this means that it plays out all the moves and gets statistics to see the best outcome), combines both measurements and moves the q-value up the tree. AlphaGo does this until it runs out of time to make the next move, at which point it simply chooses the option with the highest value.

What is shown at around 35:00 is a pretty good intuition of what is going on. He shows a big search tree, the policy network reduces its width since it makes it only evaluate 3 moves instead of all of them; the value network reduces its depth, giving you the outcome without evaluating all the tree.

Later, he explains that the computing power is equivalent to around 50-100 GPUs. That it is hard to paralleliza the Monte-Carlo Tree Search, that it doesn't get much better with more power, and that the distributed version only wins 70% of the games against the single-GPU version.

At 43:22 he shows differences between Deep Blue (the chess playing machine) and AlphaGo. The most interesting is that DeepBlue calculated 200 million positions per second, but AlphaGo only 100,000 per second.

#### [Video: Inside DeepMind](https://www.youtube.com/watch?v=xN1d3qHMIEQ&index=7&list=WL)

A short video where nature video interviews key people inside DeepMind's offices in London. At first I thought it would be worthless, but in around the 4 minute mark a few insights are given. What I thought was interesting was that one of the problems of the algorithm was about long term planning, since it had no real memory to remember the past. I thought that RNNs are supposed to be good at that, and they probably thought of it, so I wonder what were the results, or if they have solved it already how they did it, since the video is from February 2015.

There is a funny part near 6:33, where a guy is playing pool, and he sucks so bad that the white ball flies off the table when he hits it. And at the very end at 8:10, another guy playing pool throws the white ball down a hole, although in his defense, he sucks less than the first one. I bet they were trying to look their best since they were getting filmed.
