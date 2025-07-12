# Perplexity in Language Models
> general meaning of perplexilit is confuse, uncertain  


Perplexity is a measurement of how well a probability model predicts a sample. In the context of Large Language Models (LLMs), it measures the model's uncertainty or "confusion" when predicting the next word in a sequence.

* **Low Perplexity**: The model is confident and less "confused" about its predictions. This indicates a better model.
* **High Perplexity**: The model is uncertain and more "confused" about its predictions. This indicates a weaker model.

---

## The Challenge with Sentence Probability

LLMs calculate the probability of a sentence by multiplying the probabilities of each word in sequence, based on the words that came before it. This is known as the chain rule of probability.

For a sentence like "a red fox.", the probability is:

$P(\text{"a red fox."}) = P(\text{"a"}) \times P(\text{"red"} | \text{"a"}) \times P(\text{"fox"} | \text{"a red"}) \times P(\text{"."} | \text{"a red fox"})$

Here, $P(\text{"red"} | \text{"a"})$ means "the probability of the word 'red' appearing, given that the previous word was 'a'."

###  Problem: Longer Sentences are Penalized

A key issue arises because probabilities are always numbers between 0 and 1. When you multiply numbers that are less than one, the result gets progressively smaller.

* `0.9 * 0.9 = 0.81`
* `0.9 * 0.9 * 0.9 * 0.9 = 0.6561` (The value shrinks closer to zero)

This means that a longer sentence will almost always have a lower raw probability than a shorter one, even if the model is very confident about each word prediction.

**Example:**
Assume for simplicity that every word has a probability of 0.5.

* **A 2-word sentence**: $P(\text{sentence}) = 0.5 \times 0.5 = 0.25$
* **A 3-word sentence**: $P(\text{sentence}) = 0.5 \times 0.5 \times 0.5 = 0.125$

The 3-word sentence appears less likely, which makes it difficult to compare models or sentence likelihoods fairly.

---

## The Solution: Normalization with Geometric Mean

To solve this, we need to normalize the probability to account for the sentence length. We do this using the **geometric mean**.

The geometric mean finds the typical value in a set of numbers by multiplying them and then taking the *n*-th root, where *n* is the number of items.

$$ \text{Geometric Mean} = \sqrt[n]{x_1 \times x_2 \times \dots \times x_n} $$

By applying this to our sentence probabilities, we are essentially calculating the average per-word probability in a way that respects the multiplicative nature of the calculation.

**Applying the Solution:**
Using our previous example where each word's probability is 0.5:

* **For the 2-word sentence:**
    $$ \sqrt[2]{0.5 \times 0.5} = \sqrt[2]{0.25} = 0.5 $$
* **For the 3-word sentence:**
    $$ \sqrt[3]{0.5 \times 0.5 \times 0.5} = \sqrt[3]{0.125} = 0.5 $$

After normalization, both sentences have the same score, allowing for a fair comparison. This normalized probability reflects the model's average confidence per word, regardless of sentence length.

---

## From Normalized Probability to Perplexity

Perplexity is directly related to this normalized probability. Formally, perplexity is the inverse of the normalized probability, raised to the power of itself. A simpler way to think about it is that **perplexity is the inverse of the geometric mean of the word probabilities**.

$$ \text{Perplexity} = \frac{1}{\sqrt[n]{P(\text{sentence})}} $$

Let's say for the sentence "a red fox.", the model calculates a normalized probability (geometric mean) of **0.465**.

$$ \text{Perplexity} = \frac{1}{0.465} \approx 2.15 $$

This perplexity value of 2.15 can be interpreted as the model being, on average, as confused as if it had to choose between approximately 2.15 words at each step. The lower this number, the better the model.