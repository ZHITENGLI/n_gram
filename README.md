# Introduction

## Attention Please

We use the gradient descent algorithm in our program, and because of its high precision, our program can run slowly for about 6 to 8 minutes.

## n-gram language model with smoothing methods

We have 8 code blocks, including the library block, three smoothing methods class blocks and the corresponding training and testing blocks.

## Library block

We introduced the math library to calculate the PPL and some gradients.

## Lidstone Smoothing block

We encapsulate Lidstone Smoothing method into a class, which contains 4 attributes and 2 methods. Although one of its attributes is dev_data, you can also pass the test_data to calculate its PPL. 

## Jelinek-Mercer Smoothing

Similar to Lidstone Smoothing, we encapsulate the interpolation smoothing into a class, which contains 5 attributes and 2 methods. Unlike Lidstone Smoothing, we need to train 2 parameters lambda_1 and lambda_2 on the dev_set.

## Good Turing Discounting

Unlike the previous two smoothing methods, Good Turing Discounting uses 2-gram instead of 3-gram, but you can easily expand it to 3-gram, just pass the corresponding parameters and nothing to change in the class.

# Test

## Preprocessing

First of all, we need to preprocess the data from train_set, dev_set and test_set. We choose to store the long word string into a list, and each entry contains a word. Then, we insert the beginning and end symbols to the article.

## Test the Lidstone Smoothing algorithm

At the very beginning, we should record 3-word phase with its prefix. 
Next, you need to assign the value of parameter k when creating a Lidstone_smoothing instance. 
Then, call the smoothing function with train_data, which is a hash list with two words prefix keys and three words value classified by its prefix. After that, you can call calculate_PPL to get the PPL of the test set.

## Test the Jelinek-Mercer Smoothing algorithm

At the very beginning, we need to record 1-word, 2-word and 3-word phase with their probability.
Second, you need to assign the initial value of lambda_1 and lambda_2 when creating a Interpolation_smoothing instance. 
Third, you can call the function train_paras to train parameters lambda_1 and lambda_2 by gradient descent. Functions gd2 and gd1 are calculated to be the derivative of PPL w.r.t. lambda_1 and lambda_2. Finally, you are supposed to call calculate_PPL to get the PPL of the test set.

## Test the Good Turing Smoothing algorithm

First, you need to assign the parameter count to determine in what range you want to apply Good Turing Discounting. Second, you are supposed to call discounting method to reassign the probability. Finally, the same as before, you can call calculate_PPL to get the PPL of the test set.
