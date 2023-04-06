def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances
    return distances[-1]
  


# Define the list of search terms and product categories
search_terms = ['hello kitty', 'allergy medication', 'toilet paper', 'lipstick', 'hand soap']
product_categories = ['beauty product', 'medication', 'household items']

# Define a dictionary to store the matching product categories for each search term
search_term_categories = {}

# Iterate over each search term
for term in search_terms:
    # Initialize the closest category and distance to None
    closest_category = None
    closest_distance = None
    
    # Iterate over each product category
    for category in product_categories:
        # Calculate the distance between the search term and product category
        distance = levenshtein_distance(term, category)
        
        # If this is the first category or if the distance is closer than previous categories, update the closest category and distance
        if closest_category is None or distance < closest_distance:
            closest_category = category
            closest_distance = distance
    
    # Add the closest category to the dictionary
    search_term_categories[term] = closest_category

# Print the matching categories for each search term
for term, category in search_term_categories.items():
    print(f'{term} is in the {category} category')

#########################################################################################
# Import required libraries
import numpy as np
import spacy

# Load a pre-trained word embedding model
nlp = spacy.load('en_core_web_lg')

# Define the list of search terms and product categories
search_terms = ['hello kitty', 'allergy medication', 'toilet paper', 'lipstick', 'hand soap']
product_categories = ['beauty product', 'medication', 'household items']

# Define a dictionary to store the matching product categories for each search term
search_term_categories = {}

# Iterate over each search term
for term in search_terms:
    # Initialize the closest category and similarity to None
    closest_category = None
    closest_similarity = None
    
    # Iterate over each product category
    for category in product_categories:
        # Calculate the similarity between the search term and product category
        similarity = nlp(term).similarity(nlp(category))
        
        # If this is the first category or if the similarity is higher than previous categories, update the closest category and similarity
        if closest_category is None or similarity > closest_similarity:
            closest_category = category
            closest_similarity = similarity
    
    # Add the closest category to the dictionary
    search_term_categories[term] = closest_category

# Print the matching categories for each search term
for term, category in search_term_categories.items():
    print(f'{term} is in the {category} category')
###############################################################################################

# Import required libraries
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Load a pre-trained language model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Define the list of search terms and product categories
search_terms = ['hello kitty', 'allergy medication', 'toilet paper', 'lipstick', 'hand soap']
product_categories = ['beauty product', 'medication', 'household items']

# Define a dictionary to store the matching product categories for each search term
search_term_categories = {}

# Iterate over each search term
for term in search_terms:
    # Initialize the closest category and similarity to None
    closest_category = None
    closest_similarity = None
    
    # Iterate over each product category
    for category in product_categories:
        # Calculate the similarity between the search term and product category using BERT embeddings
        inputs = tokenizer.encode_plus(term, category, add_special_tokens=True, return_tensors='pt')
        outputs = model(**inputs)[1]
        similarity = np.inner(outputs.detach().numpy(), outputs.detach().numpy())[0][1]
        
        # If this is the first category or if the similarity is higher than previous categories, update the closest category and similarity
        if closest_category is None or similarity > closest_similarity:
            closest_category = category
            closest_similarity = similarity
    
    # Add the closest category to the dictionary
    search_term_categories[term] = closest_category

# Print the matching categories for each search term
for term, category in search_term_categories.items():
    print(f'{term} is in the {category} category')
