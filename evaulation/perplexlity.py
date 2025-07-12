import math

# Probabilities for Sentence A: "A red fox."
probs_a = [0.4, 0.27, 0.55, 0.79]
# Probabilities for Sentence B: "The quick brown fox jumps."
probs_b = [0.5, 0.3, 0.4, 0.6, 0.7]

# --- Calculate Raw Probability ---

# Using math.prod() to multiply all elements in the list
raw_prob_a = math.prod(probs_a)
raw_prob_b = math.prod(probs_b)

print(f"--- Raw Probabilities ---")
print(f"Sentence A: 'A red fox.'")
print(f"Individual Probabilities: {probs_a}")
print(f"Raw Probability: {raw_prob_a:.4f}\n")

print(f"Sentence B: 'The quick brown fox jumps.'")
print(f"Individual Probabilities: {probs_b}")
print(f"Raw Probability: {raw_prob_b:.4f}\n")


# --- Normalize using Geometric Mean ---

# Number of words (tokens) in each sentence
n_a = len(probs_a)
n_b = len(probs_b)

# Calculate the normalized probability (geometric mean)
normalized_prob_a = raw_prob_a**(1/n_a)
normalized_prob_b = raw_prob_b**(1/n_b)

print(f"--- Normalized Probabilities ---")
print(f"Sentence A has {n_a} words.")
print(f"Normalized Probability (Geometric Mean): {normalized_prob_a:.4f}\n")

print(f"Sentence B has {n_b} words.")
print(f"Normalized Probability (Geometric Mean): {normalized_prob_b:.4f}\n")

# --- Conclusion ---
print(f"--- Comparison ---")
if normalized_prob_b > normalized_prob_a:
    print("After normalization, Sentence B is considered more probable on a per-word basis.")
else:
    print("After normalization, Sentence A is considered more probable on a per-word basis.")