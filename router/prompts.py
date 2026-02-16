# Standardized benchmark prompts for model profiling.
# Inspired by MT-Bench and other common LLM evaluation datasets.

BENCHMARK_PROMPTS = {
    "reasoning": [
        "If you have three apples and you give one to a friend, how many apples do you have left? Explain your reasoning.",
        "Suppose you have a 5-liter jug and a 3-liter jug. How can you get exactly 4 liters of water? Provide step-by-step instructions.",
        "All humans are mortal. Socrates is a human. Therefore, Socrates is mortal. What is this type of logical argument called, and is it valid?",
        "If a plane crashes on the border of the United States and Canada, where do they bury the survivors?",
        "Imagine a world where time flows backwards. Describe a typical morning routine for someone living in this world."
    ],
    "coding": [
        "Write a Python function to check if a string is a palindrome. Include docstrings and an example.",
        "Explain the difference between a list and a tuple in Python. Provide a code example for each.",
        "Write a SQL query to find the second highest salary from an 'Employees' table with columns 'id' and 'salary'.",
        "Create a simple React functional component that displays a 'Hello, World!' message and a button that increments a counter.",
        "What is the time complexity (Big O notation) of a binary search algorithm on a sorted list of N elements? Explain why."
    ],
    "creativity": [
        "Write a short poem (8-12 lines) about a lonely robot exploring a deserted city.",
        "Brainstorm five unique and creative names for a new coffee shop that also sells vintage books.",
        "Write the first paragraph of a science fiction story that begins with the sentence: 'The stars were flickering in a way that didn't seem natural.'",
        "Imagine you are a tour guide for a city located entirely underwater. Describe the most popular tourist attraction in that city.",
        "Create a humorous dialogue between a cat and a dog who are secretly planning a world-record attempt for the most synchronized nap."
    ]
}
