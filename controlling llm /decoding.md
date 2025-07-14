# 📝 Controlling LLM Decoding: Simple Notes

## 1️⃣ Greedy Decoding

At each step, the LLM just picks the word with the highest probability right then and there. 

**Example:**  
Let's say the LLM sees **"The cat sat on the..."** and its probabilities for the next word are:

- "mat": 0.9  
- "rug": 0.08  
- "dog": 0.02

Greedy decoding will always pick "mat" because it has the highest probability (0.9).

**Advantages:**
- **Fast:** Super quick because it only makes one decision at a time.
- **Deterministic:** If you give it the same input, you'll always get the exact same output.

**Disadvantages:**
- Can miss the best overall sentence.

- Not very creative: The output can sound dull or generic.
- Can be repetitive


>Path A:

 P("A") * P("big" | "The") * P("red" | "The big")
= 0.9 * 0.5 * 0.6 = 0.27

>Path B:

P("The") * P("large" | "The") * P("blue" | "The large")
= 0.5* 0.9 * 0.8 = 0.36


here , since the probability 
p(A) is 0.9 it choose path a 
which is , [A big red] which total probability is 0.27 which is less than 0.36 of path b 

so this is main disadvantage of being greedy 



## 2️⃣ Beam Search
(a maze but keeping track of a few of the most promising paths at the same time)


> Instead of just picking the single most probable word, beam search keeps track of the top 'B' (beam width) most probable sequences at each step. 'B' is a number you choose.

**Example:**  
Let B = 2.

- **Step 1:**  
"The" (0.8)
"A" (0.7)
"My" (0.1)

Keeps "The" and "A".


- **Step 2:**  
From "The":
"The cat" = 0.8 * 0.7 = 0.56
"The dog" = 0.8 * 0.6 = 0.48

From "A":

"A cat" = 0.7 * 0.8 = 0.56

"A bird" = 0.7 * 0.5 = 0.35



Picks the top 2: "The cat" and "A cat".

**Advantages:**
- Better quality: Often more coherent.
- More likely to find a good sequence.

**Disadvantages:**
- Slower, more resource-intensive.
- Can still be repetitive.
- No guarantee of the absolute best sequence.

---

## 3️⃣ Sampling Methods (Adding Randomness)

Sometimes you want the LLM to be more creative and less predictable.

---

### 🔥 a) Temperature Sampling

**"Temperature"** controls how "daring" the LLM is allowed to be.

**How it works:**  
It adjusts the "sharpness" of the probability distribution.

- **Low temperature (0.1 - 0.5):** Makes the highest probability words even more likely. Close to greedy.
- **Medium (0.6 - 0.8):** Good balance.
- **High (1.0 - 1.5):** Flattens the distribution, giving unlikely words more chance.

**Numerical Explanation:**  
Imagine the logits (raw scores) get divided by T.

