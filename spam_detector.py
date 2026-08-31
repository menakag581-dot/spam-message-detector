from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample messages
messages = [
    "Win a free prize now",
    "Congratulations you won a lottery",
    "Claim your free gift",
    "You have won free money",
    "Are you coming to college today",
    "Please send me the notes",
    "Let's meet tomorrow",
    "Can you call me later"
]

# Labels for the messages
labels = [
    "spam",
    "spam",
    "spam",
    "spam",
    "not spam",
    "not spam",
    "not spam",
    "not spam"
]

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Create and train the model
model = MultinomialNB()
model.fit(X, labels)

# Get message from user
message = input("Enter a message: ")

# Convert the new message into numbers
message_vector = vectorizer.transform([message])

# Predict
prediction = model.predict(message_vector)

# Display result
print("\nResult:", prediction[0])
