# NOTA : We call an array (containing 0-digits and 1-digits) a "vector" through misuse of language.
# A vector digit indicates the recommendation or not of a pair action-resource (observing the same index in df). For example, a vector containing "1,0,0,1,1,0,0,1,0,1,1,0,1..." means that the first action-resource in df is recommended (because first digit is a 1-digit) but not the second one (because second digit is a 0-digit), and so on.


# Vector Score
# Function to evaluate the score of a vector.
def score(v): 
  score = 0
  for i in range(len(v)):
    if v[i] > 0: # =1 most of the time
      for weight_index in range(0, len(weights)):
        score += weights[weight_index] * df[i][weight_index+1]
  return score


# Resource Score
# Function to evaluate the score of a specific resource.
def resource_score(resource_index): 
  score = 0
  for weight_index in range(0, len(weights)):
    score += weights[weight_index] * df[resource_index][weight_index+1]
  return score


# Vector Constraints Completion Score
# Function to evaluate the constraints completion score of a vector.
def vector_constraints_score(v):
  v_cc_score = 0
  for k in range(len(v)):
    if v[k] == 1:
      v_cc_score += resource_constraints_score(k)
  return v_cc_score


# Resource Constraints Completion Score
# Function to evaluate the mean constraints completion score of a specific resource using each constraint max_value. The percentage of completion is calculated toward these max_values.
def resource_constraints_score(resource_index): 
  scores = []
  for k in range(len(constraints)):
    # Constraint Score percentage
    score = df[resource_index][constraints[k][0]]/constraints[k][1]
    scores.append(score)
  mean_score = mean(scores) # mean constraints completion score
  return mean_score
  

# Constraints checker
# Function to check if a vector respects all min_value constraints and all max_value constraints.
def constraints_check(v):
  # Initialise v_constraints_values.
  v_constraints_values = []
  for k in range(len(constraints)):
    v_constraints_values.append(0)
  
  # Feed v_constraints_values.
  for d in range(len(v)):
    if v[d] > 0: # =1 most of the time
      for k in range(len(constraints)):
        constraint_index_in_df = constraints[k][0]
        v_constraints_values[k] += df[d][constraint_index_in_df]
        
  # Check if v_constraints_values respect constraints.
  respect = True
  for k in range(len(constraints)):
    if v_constraints_values[k] > constraints[k][1]: # Above max_value
      respect = False
    if v_constraints_values[k] < constraints[k][2]: # Below min_value
      respect = False
      
  return respect


# Constraints min_values checker
# Function to check if a vector respects all min_value constraints.
def constraints_min_values_check(v):
  # Initialise v_constraints_values.
  v_constraints_values = []
  for k in range(len(constraints)):
    v_constraints_values.append(0)
  
  # Feed v_constraints_values.
  for d in range(len(v)):
    if v[d] > 0: # =1 most of the time
      for k in range(len(constraints)):
        constraint_index_in_df = constraints[k][0]
        v_constraints_values[k] += df[d][constraint_index_in_df]
        
  # Check if v_constraints_values respect all min_value constraints.
  respect = True
  for k in range(len(constraints)):
    if v_constraints_values[k] < constraints[k][2]: # Below min_value
      respect = False
      
  return respect
  
  
# Constraints max_values checker
# Function to check if a vector respects all max_value constraints.
def constraints_max_values_check(v):
  # Initialise v_constraints_values.
  v_constraints_values = []
  for k in range(len(constraints)):
    v_constraints_values.append(0)
  
  # Feed v_constraints_values.
  for d in range(len(v)):
    if v[d] > 0: # =1 most of the time
      for k in range(len(constraints)):
        constraint_index_in_df = constraints[k][0]
        v_constraints_values[k] += df[d][constraint_index_in_df]
        
  # Check if v_constraints_values respect all max_value constraints.
  respect = True
  for k in range(len(constraints)):
    if v_constraints_values[k] > constraints[k][1]: # Above max_value
      respect = False
      
  return respect



  

# Generate Vector
# Function to create a vector containing randomly placed 1-digits.
def generate_vector():
  v = []
  for m in range(len(df)):
    v.append(0)

  # Get 0-digits indexes and 1-digits indexes from v.
  zero_digits_indexes = []
  one_digits_indexes = []
  for d in range(len(v)):
    if v[d] == 0:
      zero_digits_indexes.append(d)
    elif v[d] == 1:
      one_digits_indexes.append(d)

  # While v respect max_value constraints...
  while constraints_max_values_check(v) is True and len(zero_digits_indexes) > 0:
    # ...Randomly transform a 0-digit into a 1-digit in v.
    rand = random.choice(zero_digits_indexes)
    v[rand] = 1
    zero_digits_indexes.remove(rand)
    one_digits_indexes.append(rand)
    
  """
  # While vector has no 1-digit (we want it to have at least one 1-digit).
  while sum(vector) == 0:  
    respect = True
    
    # Security counter to avoid being trapped into an infinite loop.
    counter_trapped = 0
    max_counter_trapped = 20
    # Max number of resets authorized.
    max_resets = 3
    
    # While vector respects constraints.
    while respect is True:
      
      # If the generation has been trapped 20 times, we reset the counter and the vector.
      if counter_trapped >= max_counter_trapped and max_resets <= 3:
        counter_trapped = 0
        for m in range(len(vector)):
          vector[m] = 0
        max_resets += 1
      else:
        respect = False
      
      # A random 0-digit becomes a 1-digit.
      rand = random.randint(0, len(df)-1)
      if vector[rand] == 0:
        vector[rand] = 1
        if constraints_max_values_check(vector) is False:
          # Cancel the modification just done.
          vector[rand] = 0
          if constraints_min_values_check(vector) is True:
            # Stop while loop.
            respect = False
          else:
            counter_trapped += 1
  """
  
  return normalise(v)


# Normalise
# Function to operate mutations on a vector in order to make it respect constraints.
def normalise(v):
  
  # Get 0-digits indexes and 1-digits indexes from v.
  zero_digits_indexes = []
  one_digits_indexes = []
  for d in range(len(v)):
    if v[d] == 0:
      zero_digits_indexes.append(d)
    elif v[d] == 1:
      one_digits_indexes.append(d)

  # Normalise.
  while constraints_max_values_check(v) is False or constraints_min_values_check(v) is False:
    # While v doesn't respect max_value constraints...
    while constraints_max_values_check(v) is False:
      # ...Randomly transform a 1-digit into a 0-digit in v.
      rand = random.choice(one_digits_indexes)
      v[rand] = 0
      one_digits_indexes.remove(rand)
      zero_digits_indexes.append(rand)

    # While v doesn't respect min_value constraints...
    while constraints_min_values_check(v) is False:
      # ...Randomly transform a 0-digit into a 1-digit in v.
      rand = random.choice(zero_digits_indexes)
      v[rand] = 1
      zero_digits_indexes.remove(rand)
      one_digits_indexes.append(rand)
  
  """
  # While v respect max_value constraints...
  while constraints_max_values_check(v) is True:
    # ...Randomly transform a 0-digit into a 1-digit in v.
    rand = random.choice(zero_digits_indexes)
    v[rand] = 1
    zero_digits_indexes.remove(rand)
    one_digits_indexes.append(rand) 
  """

  """
  # NOTA : The code below is really powerfull to increase the score. I decided to remove it from the normalise() function because it is not the role of the normalise() function to do that. I use it instead in the function trying to guide exploration and exploitation. Indeed this code is prone to introduce new 1-digits which is good for the exploration part.
  # Try random upgrades until constraints_max_values is no more respected.
  ok = True
  while ok is True:
    # A random 0-digit become a 1-digit.
    rand = random.choice(zero_digits_indexes)
    v[rand] = 1
    zero_digits_indexes.remove(rand)
    one_digits_indexes.append(rand)
    if constraints_max_values_check(v) is False:
      # End main while loop.
      ok = False
      # Cancel the mutation just done.
      v[rand] = 0
      one_digits_indexes.remove(rand)
      zero_digits_indexes.append(rand)
  """
  return v


# Add To top_vectors
# Function to add a vector into top_vectors, top_vectors being limited by a size max.
# top_vectors is used to make the selection of vectors constituting actors' final recommendations.
def add_to_top_vectors(vector, score):
  global top_vectors
  v_score = score
  
  # If top_vectors is empty.
  if len(top_vectors) == 0:
    # Add vector to top_vectors.
    top_vectors.append({'vector':vector, 'score':v_score})
  # Else if max size of top_vectors is not reached.
  elif len(top_vectors) < max_top_vectors:  
    # Check if vector already exists in top_vectors.
    already_exist = False
    for v in range(0, len(top_vectors)):
      common_digits = 0
      for d in range(0, len(vector)):
        if top_vectors[v]['vector'][d] == 1 and vector[d] == 1:
          common_digits += 1
      if common_digits == sum(top_vectors[v]['vector']): 
        already_exist = True
        break
    if already_exist is False:
      # Add vector to top_vectors.
      top_vectors.append({'vector':vector, 'score':v_score})
      # Sort top_vectors by score DESC.
      top_vectors = sorted(deepcopy(top_vectors), key=lambda dct: dct['score'], reverse=True)
  # Else if max size of top_vectors is reached and vector score is good enough to be incorporated into top_vectors (better score than the weakest score in top_vectors).
  elif v_score > top_vectors[max_top_vectors-1]['score']:
    # Check if vector already exists in top_vectors.
    already_exist = False
    for v in range(0, len(top_vectors)):
      common_digits = 0
      for d in range(0, len(vector)):
        if top_vectors[v]['vector'][d] == 1 and vector[d] == 1:
          common_digits += 1
      if common_digits == sum(top_vectors[v]['vector']): 
        already_exist = True
        break
    if already_exist is False:
      # Remove the last vector (lowest score) from top_vectors.
      top_vectors.pop(max_top_vectors-1)
      # Add vector to top_vectors.
      top_vectors.append({'vector':vector, 'score':v_score})
      # Sort top_vectors by score DESC.
      top_vectors = sorted(deepcopy(top_vectors), key=lambda dct: dct['score'], reverse=True) 


# Add to vectors_classes
# Function used to classify vectors at each iteration end in order to determine if it was an exploration or exploitation iteration.
# If a vector leads to the creation of a new class, we consider that it is an exploration vector and that the iteration where it was created is an exploration iteration. Other vectors that are just added to existing classes, are considered as exploitation vectors and associated iterations are considered as exploitation iterations.
# Vectors and classes are added into vectors_classes and E&E phases are added into ee_phases. Both arrays are declared in main.py.
def add_to_vectors_classes(vector, iteration_num):
  global vectors_classes
  global ee_phases

  # Minimum similarity percentage
  min_sim_percent = 0.5
  
  # Starting length of vectors_classes.
  vc_len_start = len(vectors_classes)
  
  # Current class to add the vector...
  retained_class = -1
  # ... and the mean sim_percent associated.
  retained_sim_percent = 0
  
  # For each class
  for c in range(len(vectors_classes)):
    sim_percents = []
    
    # For each vector
    for v in range(len(vectors_classes[c])):
      # Calculate the similarity percentage between the parameter vector and vectors_classes[c][v]. Only 1-digits are considered, concretely 1-digits in common divided by distinct 1-digits quantity on both vectors.
      sim_percent = 0
      total_of_1_digits = 0
      for d in range(len(vector)):
        if vector[d] == 1 or vectors_classes[c][v][d] == 1:
          total_of_1_digits += 1
        if vector[d] == 1 and vectors_classes[c][v][d] == 1:
          sim_percent += 1
      
      # Save the similarity percentage.
      sim_percent /= total_of_1_digits
      sim_percents.append(sim_percent)
      
    # Calculate the mean sim_percent.
    mean_sim_percent = mean(sim_percents)
    
    # If the mean_sim_percent is superior to min_sim_percent and to retained_sim_percent :
    # - save it as the new retained_sim_percent
    # - update retained_class
    if mean_sim_percent > min_sim_percent and mean_sim_percent > retained_sim_percent:
      retained_sim_percent = mean_sim_percent
      retained_class = c
  
  # Add the vector to the retained class if the latter is not -1.
  if retained_class > -1:
    vectors_classes[retained_class].append(vector)
  # Otherwise, create a new class and add the vector to it.
  else:
    vectors_classes.append([])
    vectors_classes[len(vectors_classes)-1].append(vector)
    
  # Ending length of vectors_classes.
  vc_len_end = len(vectors_classes)
  
  # If a new class was created, it is an exploration iteration. Save this information into ee_phases.
  if vc_len_end > vc_len_start:
    ee_phases[iteration_num] = "Exploration"
    




 
# Similarity
# Function to return the lowest, mean and highest similarity percentage of multiple vectors compared with a reference vector. 
# The reference vector is the strongest vector and is positioned in index 0 of vectors.
# Doing that, we give priority to the score in the first place.
def check_similarity(vectors):
  # Determine the lowest, mean and highest similarity.
  lowest_similarity = 1
  highest_similarity = 0
  similarity_values = []
  for k in range(1, len(vectors)):
    similar_1_digits = 0
    for d in range(0, len(vectors[0]['vector'])):
      if vectors[0]['vector'][d] == 1 and vectors[k]['vector'][d] == 1:
        similar_1_digits += 1
    similarity = similar_1_digits/sum(vectors[0]['vector'])
    similarity_values.append(similarity)
    if similarity < lowest_similarity:
      lowest_similarity = similarity
    if similarity > highest_similarity:
      highest_similarity = similarity
  # If there is only one vector in vectors, the above for loop will be skipped and similarity_values will be empty. In this case, all similarities are considered to be 100%.
  if len(similarity_values) > 0:
    mean_similarity = mean(similarity_values)
  else:
    lowest_similarity = 1
    highest_similarity = 1
    mean_similarity = 1
  return [lowest_similarity, highest_similarity, mean_similarity]

  
# Display Recommendations
# Function to select a vector for each concerned actor and display associated recommendations. 
# The process tries to maximize the similarity between selected vectors (if this part of the code is activated).
# NOTA : Parameters iteration_score_evolution and max_score_evolution are used to plot the evolution of the algorithm through iterations, to see the natural evolutionist convergence behaviour. So, final recommendations displayed here don't correspond in all cases to this evolution because they are done by checking the similarity which can lead to ignore some vectors.
def select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration):  
  global top_vectors
  global all_vectors

  """
  # Plot max_score_evolution
  fig = plt.figure(figsize=(8,5))
  ax = fig.add_axes([0.1, 0.1, 0.6, 0.7])
  fig.suptitle('P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_Max_Score_Evolution', fontsize=12)
  plot_range = [*range(0, iteration_number, 1)]
  ax.plot(plot_range, max_score_evolution, 'y-', linewidth=0.5, 
            label='Max Score Evolution')
  ax.set_xlabel('Iteration', size=10)
  ax.set_ylabel('Value', size=10)
  ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.show() 
  plt.clf()
  plt.close('all') 
  """
  
  # Firstly, select the x first vectors in top_vectors, x being actors_number.
  selected_vectors_bests = top_vectors[:actors_number]
  
  """
  # SIMILARITY - ACTIVATE IF NEEDED
  sims = check_similarity(selected_vectors_bests)
  lowest_similarity_bests = sims[0]
  highest_similarity_bests = sims[1]
  mean_similarity_bests = sims[2]
  final_lowest_similarity = lowest_similarity_bests
  final_highest_similarity = highest_similarity_bests
  final_mean_similarity = mean_similarity_bests
  """
  
  final_vectors_selection = selected_vectors_bests
  sum_scores = 0
  
  
  # MEAN SCORE AMONG VECTORS RECOMMENDED TO ACTORS
  for k in range(0, actors_number):
    sum_scores += final_vectors_selection[k]['score']
  final_mean_score = sum_scores/actors_number
  
  
  """
  # SIMILARITY - ACTIVATE IF NEEDED
  # If the lowest percentage of similar resources is below the required percentage, we try to improve it by making another selection.
  # Doing that, we will give priority to the similarity to make a new vector selection among top_vectors.
  if lowest_similarity_bests < similarity_percentage:
    # Select another vectors combination by keeping the strongest vector and the other ones having the best similarity with it.
    selected_vectors = []
    # Keep the strongest vector.
    selected_vectors.append(top_vectors[0])
    # Rank other vectors.
    ranks = []
    for k in range(1, len(top_vectors)):
      sims = check_similarity([top_vectors[0], top_vectors[k]]) # As we call similarity() with only two vectors, the function will return the same value for lowest and highest similarity ...
      ranks.append({
                    'index_in_top_vectors':k, 
                    'similarity':sims[0] # ... so we put here index 0 or 1 (same values)
                  })
    # Sort ranks by similarity DESC.
    ranks = sorted(deepcopy(ranks), key=lambda dct: dct['similarity'], reverse=True)
    # Append the best vectors having the best similarity with the strongest vector into selected_vectors respecting the number of actors.
    for k in range(1, actors_number):
      selected_vectors.append(top_vectors[ranks[k-1]['index_in_top_vectors']])
    
    # If the new lowest similarity of the new selected vectors is superior to the first lowest similarity calculated, then we keep the new selection, otherwise we keep the first selection.
    if ranks[actors_number-2]['similarity'] > lowest_similarity_bests:
      final_lowest_similarity = ranks[actors_number-2]['similarity']
      final_highest_similarity = ranks[0]['similarity']
      
      sum_similarities = 0
      for k in range(0, actors_number-1):
        sum_similarities += ranks[k]['similarity']
      final_mean_similarity = sum_similarities/(actors_number-1)
        
      final_vectors_selection = selected_vectors
      
      sum_scores = 0
      for u in range(0, actors_number):
        sum_scores += final_vectors_selection[u]['score']
      final_mean_score = sum_scores/actors_number
  
  # It is important to note that in the end, the selection reflects one of these 3 cases (always keeping the best score vector) :
  # Selection with the best scores and with similarity cap respected.
  # Selection with the best similarity but potentially with inferior scores.
  # Selection with the best scores and with similarity cap not respected because it is impossible to improve the similarity.
  """

  
  # Presence Rate
  # Create a vector named dense_vector containing for each digit of all_vectors the number of times where it has value 1.
  dense_vector = sum(all_vectors, axis=0)
  # Create a variable named presence_rate being the mean of dense_vector values divided by the number of vectors in all_vectors.
  # This is the mean presence rate of digits among all vectors.
  # The more this number is low, the more digits are absent in a lot of vectors and so the more the research was centered on exploration. 
  # The more this number is high, the more digits are present in a lot of vectors and so the more the research was centered on exploitation. 
  presence_rate = round(mean(dense_vector) / len(all_vectors), 1)
  
  
  
  # Presence Gap
  # Create a vector named presence_gaps containing for each digit of dense_vector its mean difference toward all other digits.
  presence_gaps = []
  for ref in range(0, len(dense_vector)):
    mean_diff_toward_all = 0
    for comp in range(0, len(dense_vector)):
      mean_diff_toward_all += abs(dense_vector[ref] - dense_vector[comp])
    mean_diff_toward_all = deepcopy(mean_diff_toward_all)/(len(dense_vector)-1)
    presence_gaps.append(round(mean_diff_toward_all, 1))
  # Create the final variable named presence_gap being the mean of presence_gaps values divided by the number of vectors in all_vectors.
  # This is the digits mean presence gap among all_vectors.
  # The more this number is low, the more digits are present in a lot of vectors and so the more the research was centered on exploitation.
  # The more this number is high, the more digits are absent in a lot of vectors and so the more the research was centered on exploration.
  #   presence_gap = round(mean(presence_gaps) / len(all_vectors), 1)
  
  
  
  # Coverage
  # Coverage definition : As input you are given several sets and a number k. The sets may have some elements in common. You must select at most k of these sets such that the maximum number of elements are covered, i.e. the union of the selected sets has maximal size.
  # Therefore we can define :
  # Global Coverage as the number of different 1-digits among all_vectors divided by the total number of digits in a vector.
  # Coverage in best solutions (in top_vectors) as the same thing but among top_vectors. 
  one_digits_in_all_vectors = 0
  for k in range(len(dense_vector)):
    if dense_vector[k] > 0:
      one_digits_in_all_vectors += 1
  all_vectors_coverage = one_digits_in_all_vectors / len(dense_vector)
  
  top_vectors_without_scores = []
  for k in range(len(top_vectors)):
    top_vectors_without_scores.append(top_vectors[k]['vector'])
  dense_top_vectors = sum(top_vectors_without_scores, axis=0)
  one_digits_in_top_vectors = 0
  for k in range(len(dense_top_vectors)):
    if dense_top_vectors[k] > 0:
      one_digits_in_top_vectors += 1
  top_vectors_coverage = one_digits_in_top_vectors / len(dense_top_vectors)
  
  
  
  # Popularity
  # The popularity value is here the mean percentage of popular items recommended among all resources recommended in all final recommendations.
  # Some items (resources) dominate the others regarding popularity which can lead the recommender to mostly suggest popular items. This is not a problem but it is interesting to get this information which can allow us to judge the quality of the recommendations proposed. 
  # We could also use this value to tune the recommendations with more or less popular items. => We don't do that here, we just calculate the popularity value for comparison purposes.
  # Let's do it :
  popularity = 0
  popularities = []
  for k in range(len(final_vectors_selection)):
    v_popularity = 0
    sum_popular = 0
    for d in range(len(final_vectors_selection[k]['vector'])):
      if final_vectors_selection[k]['vector'][d] == 1 and d in popular_items_indexes:
        sum_popular += 1
    v_popularity = sum_popular / sum(final_vectors_selection[k]['vector'])
    popularities.append(v_popularity)
  popularity = mean(popularities)
  
  
  
  # Constraints completion percentage regarding max_values.
  # Get all constraints values.
  all_constraints_values = []
  for k in range(0, len(final_vectors_selection)):  
    
    vector_constraints_values = []
    for c in range(len(constraints)):
      vector_constraints_values.append(0)
    
    for i in range(len(final_vectors_selection[k]['vector'])):
      if final_vectors_selection[k]['vector'][i] == 1:
        for c in range(len(constraints)):
          vector_constraints_values[c] += df[i][constraints[c][0]]
    all_constraints_values.append(vector_constraints_values)
  # Calculate the mean of each constraints from all_constraints_values.
  # Calculate the percentage of this mean toward the constraint maximum value.
  # Stock the percentages into mean_constraints_percentages.
  mean_constraints_percentages = []
  transposed_values = transpose(array(all_constraints_values))
  for k in range(len(transposed_values)):
    percentage = mean(transposed_values[k])/constraints[k][1]
    mean_constraints_percentages.append(percentage)
  # Get the global mean of constraints percentages, this is our global constraints completion.  
  global_constraints_completion = mean(mean_constraints_percentages)
  

  
  # Potential
  # For each vector in final_vectors_selection, calculate its potential.
  potentials = []
  for p in range(len(final_vectors_selection)):
    potential = 0
    for d in range(len(best_worse_digits)):
      # If the vector value is not the wanted value, we increment its potential.
      if final_vectors_selection[p]['vector'][best_worse_digits[d]['index']] != best_worse_digits[d]['wanted_value']:
        potential += 1
    # Final potential as a percentage.
    potential /= len(best_worse_digits)
    # Append to potentials
    potentials.append(potential)
  # Calculate the mean potential of final_vectors_selection.
  mean_potential = mean(potentials)
  
  
  
  # Links Gap
  # Initialise the matrix which will contain the number of links between each digits couple in all_vectors.
  links = [[0 for i in range(len(all_vectors[0]))] for j in range(len(all_vectors[0]))]
  # Get 1-digit indexes from vectors in all_vectors.
  one_digits_indexes = []
  for k in range(len(all_vectors)):
    vector_one_digits_indexes = []
    for d in range(len(all_vectors[k])):
      if all_vectors[k][d] == 1:
        vector_one_digits_indexes.append(d)
    one_digits_indexes.append(vector_one_digits_indexes)
  # In each vector, for each digits couple possible, increment the link value in links.
  for k in range(len(one_digits_indexes)):
    for index in range(len(one_digits_indexes[k])):
      for index2 in range(index+1, len(one_digits_indexes[k])):
        links[one_digits_indexes[k][index]][one_digits_indexes[k][index2]] += 1
  # Transform the links matrix into an array.
  links = array(links).flatten()
  # Remove zero values from links.
  links = links[links != 0]
  # Let's calculate our indicator named links_gap.
  # Calculate the gap between the max and the mean number of links.
  # Calculate how many percents the gap represents toward the max number of links.
  mean_number_of_links = mean(links)
  max_number_of_links = max(links)
  gap = max_number_of_links - mean_number_of_links
  links_gap = gap/max_number_of_links
  
  
  
  # Vector Links
  # Initialise the array which will contain the number of links in each vector from all_vectors.
  links = []
  # Get the number of 1-digits in each vector, calculate the number of link possibilities and append it to links.
  for k in range(len(all_vectors)):
    v_one_digit_number = sum(all_vectors[k])
    link_possibilities = (v_one_digit_number*(v_one_digit_number-1))/2
    links.append(link_possibilities)
  # Calculate our indicator named vector_links_gap.
  # Calculate the gap between the max and the mean of links values.
  # Calculate how many percents the gap represents toward the max of links values.
  mean_links = mean(links)
  max_links = max(links)
  gap = max_links - mean_links
  vector_links_gap = gap/max_links
  

  
  # ------------------------------
  # E&E POWER
  # ------------------------------
  # pexplor = exploration power
  # pexploit = exploitation power
  # ------------------------------
  # counter_explore_iter 
  # = number of iterations where the size of "ndd" has diminished. This corresponds to the number of iterations centered on exploration.
  # ------------------------------
  # counter_exploit_iter 
  # = number of iterations where the size of "ndd" has not diminished. This corresponds to the number of iterations centered on exploitation.
  # ------------------------------
  # counter_explore_vector 
  # = sum on all iterations of the number of vectors containing digits from "ndd". This corresponds to the number of vectors that participated to exploration.
  # ------------------------------
  # counter_exploit_vector 
  # = sum on all iterations of the number of vectors not containing digits from "ndd". This corresponds to the number of vectors that participated to exploitation.
  # ------------------------------
  # counter_explore_digit
  # = sum on all iterations of newly discovered digits.
  # ------------------------------
  # counter_exploit_digit
  # = sum on all iterations of already discovered digits.
  # ------------------------------
  # Mean number of exploration vectors per exploration iteration.
  #mean_explore_v = counter_explore_vector / counter_explore_iter
  # ------------------------------
  # Mean number of exploitation vectors per exploitation iteration.
  #mean_exploit_v = counter_exploit_vector / counter_exploit_iter
  # ------------------------------
  # Mean number of newly discovered digits per exploration iteration.
  #mean_explore_d = counter_explore_digit / counter_explore_iter
  # ------------------------------
  # Mean number of already discovered digits per exploitation iteration.
  #mean_exploit_d = counter_exploit_digit / counter_exploit_iter
  # ------------------------------
  # Calculate ee_power as the mean of three percentages calculated with the six E&E power global variables.
  # NOTA : percentage_3 is multiplied by 1000 to make its value in the same height order as the two other percentages.
  percentage_1 = 1
  if counter_exploit_iter > 0:
    percentage_1 = counter_explore_iter / counter_exploit_iter
  percentage_2 = 1
  if counter_exploit_vector > 0:
    percentage_2 = counter_explore_vector / counter_exploit_vector
  percentage_3 = 1
  if counter_exploit_digit > 0:
    percentage_3 = (counter_explore_digit / counter_exploit_digit)*1000
  ee_power = mean([percentage_1, percentage_2, percentage_3])
  # ------------------------------
  
  
  
  # ------------------------------
  # E&E Temporal Balance
  # ------------------------------
  # Calculate gaps between successive timers from explore_iter_timers.
  # We do not consider the starting algorithm execution timer (saet timer) and the last algorithm execution timer (laet timer) because we consider that the most important is to analyse the core of the E&E behaviour from where it started to where it ended. We justify that by the fact that the gap between the saet and the first exploration timer could take a completely absurd value, same thing for the gap between the last exploration timer and the laet.
  e_i_timers_gaps = []
  for k in range(len(explore_iter_timers)-1):
    gap = explore_iter_timers[k+1] - explore_iter_timers[k]
    e_i_timers_gaps.append(gap)
  if len(e_i_timers_gaps) > 0:
    # Min gap
    min_eit_gap = min(e_i_timers_gaps)
    # Max gap
    max_eit_gap = max(e_i_timers_gaps)
    # Difference between min_gap and max_gap.
    diff_min_max_eit_gaps = max_eit_gap - min_eit_gap
    # Mean gap
    mean_eit_gap = mean(e_i_timers_gaps)
    # Mean difference of the mean differences between each gap and the others.
    diffs = []
    for k in range(len(e_i_timers_gaps)):
      for g in range(len(e_i_timers_gaps)):
        if g != k:
          diffs.append(abs(e_i_timers_gaps[k] - e_i_timers_gaps[g]))
    if len(diffs) > 0:
      mean_diff_between_eit_gaps = mean(diffs)
    else:
      mean_diff_between_eit_gaps = math.inf
  else:
    min_eit_gap = math.inf
    max_eit_gap = math.inf
    diff_min_max_eit_gaps = math.inf
    mean_eit_gap = math.inf
    mean_diff_between_eit_gaps = math.inf
  

  """
  # Display ee_logs.
  print("-----------------------------------")
  for k in range(len(ee_logs)):
    print("")
    print("-----------------------------------")
    print("Iteration : " + str(ee_logs[k]['iteration']))
    print("-----------------------------------")
    print("Vectors Quantity : " + str(ee_logs[k]['vectors_quantity']))
    print("Top Quantity : " + str(ee_logs[k]['top_quantity']))
    print("Hyper Volume : " + str(ee_logs[k]['hyper_volume']))
    print("Power : " + str(ee_logs[k]['power']))
    print("Density : " + str(ee_logs[k]['density']))
    print("")
    
    print("Vectors : ")
    for v in range(len(ee_logs[k]['vectors'])):
      print("")
      print(ee_logs[k]['vectors'][v]['vector'])
      print(">>> " + ee_logs[k]['vectors'][v]['ee_type'])
      print(">>> New Digits : " + str(ee_logs[k]['vectors'][v]['number_of_new_digits']))
      print(">>> Cause : " + ee_logs[k]['vectors'][v]['cause'])
      print(">>> Timer : " + str(ee_logs[k]['vectors'][v]['timer']))
    print("-----------------------------------")
    
  print("-----------------------------------")
  """
  
  
  """
  # --------------------
  # E&E Hyper-Volumic Indicators
  # --------------------  
  # E&E Multi Display.
  
  # Preparing data.
  iters = len(ee_logs)
  powers = []
  top_vectors_scores = []
  all_vectors_coverages = []
  top_vectors_constraints_completions = []
  similarities_with_top_v = []    
  hvv_values = []
  hvo_values = []
  d_v_values = [] # Dimensions values portions
  d_o_values = [] # Dimensions occurrences portions
  hvcv_values = []
  hvco_values = []  
  
  if len(ee_logs) > 0:

    for k in range(len(ee_logs)):
      powers.append(ee_logs[k]['power'])
      top_vectors_scores.append(ee_logs[k]['top_vectors_score'])
      all_vectors_coverages.append(ee_logs[k]['all_vectors_coverage'])
      top_vectors_constraints_completions.append(ee_logs[k]['top_vectors_constraints_completion'])
      similarities_with_top_v.append(ee_logs[k]['similarity_with_top_vectors'])
      
      hvv_values.append(ee_logs[k]['hvv'])
      hvo_values.append(ee_logs[k]['hvo'])
      d_v_values.append(ee_logs[k]['d%v'])
      d_o_values.append(ee_logs[k]['d%o'])
      hvcv_values.append(ee_logs[k]['hvcv'])
      hvco_values.append(ee_logs[k]['hvco'])
    
    # Scale data. Scales values to have them laying between 0 and 1.
    # Scale hvcv_values.
    max_value = max(hvcv_values)
    min_value = min(hvcv_values)
    diff = max_value - min_value 
    for k in range(len(hvcv_values)):
      hvcv_values[k] = (hvcv_values[k] - min_value)/diff
    # Scale hvco_values.
    max_value = max(hvco_values)
    min_value = min(hvco_values)
    diff = max_value - min_value 
    for k in range(len(hvco_values)):
      hvco_values[k] = (hvco_values[k] - min_value)/diff
    # Scale top_vectors_scores.
    max_bvs = max(top_vectors_scores)
    min_bvs = min(top_vectors_scores)
    diff = max_bvs - min_bvs
    for k in range(len(top_vectors_scores)):
      top_vectors_scores[k] = (top_vectors_scores[k] - min_bvs)/diff
    
    # Plot indicators evolution.
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.7])
    fig.suptitle('P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_E&E_Multi', fontsize=12)
    plot_range = [*range(0, iters, 1)]
    
    ax.plot(plot_range, powers, 'b-', linewidth=0.5, 
              label='Power')
    ax.plot(plot_range, top_vectors_scores, 'c-', linewidth=0.5, 
              label='TopV Score (scaled)')
    ax.plot(plot_range, all_vectors_coverages, 'm-', linewidth=0.5, 
              label='AllV Coverage')
    ax.plot(plot_range, top_vectors_constraints_completions, 'y-', 
              linewidth=0.5, 
              label='TopV Const Comp')
    ax.plot(plot_range, similarities_with_top_v, color='#FF9E00', 
              linewidth=0.5, 
              label='Similarity with TopV')          
              
    ax.plot(plot_range, hvv_values, 'r-', linewidth=0.5, 
              label='HVV')      
    ax.plot(plot_range, hvo_values, 'r--', linewidth=0.5, 
              label='HVO')
    ax.plot(plot_range, hvcv_values, 'g-', linewidth=0.5, 
              label='HVCV (scaled)')
    ax.plot(plot_range, hvco_values, 'g--', linewidth=0.5, 
              label='HVCO (scaled)')  
    
    ax.axvline(x=convergence_iteration, ymin=0, ymax=1000, 
              label='Convergence', color='black', linestyle='dashed', linewidth=1) # Vertical line showing convergence.
    
    ax.set_xlabel('Iteration', size=10)
    ax.set_ylabel('Value', size=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig('Experiments/Charts/Multi/Multi/P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_E&E_Multi.png', dpi=300, bbox_inches='tight')
    #plt.show() 
    plt.clf()
    plt.close('all')   
    
    # Plot d_v_values.
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.7])
    fig.suptitle('P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) +'_E&E_d_v_values', fontsize=12)
    plot_range = [*range(0, iters, 1)]
    
    transposed_d_v_values = np_matrix(d_v_values).T
    for k in range(len(transposed_d_v_values)):
      r = random.uniform(0, 1)
      g = random.uniform(0, 1)
      b = random.uniform(0, 1)
      ax.plot(plot_range, transposed_d_v_values[k].tolist()[0], 
      color=[r, g, b], linewidth=0.5, 
      label='D%V - Dim '+str(k+1))      
    
    ax.axvline(x=convergence_iteration, ymin=0, ymax=1000, 
              label='Convergence', color='black', linestyle='dashed', linewidth=1) # Vertical line showing convergence.
    
    ax.set_xlabel('Iteration', size=10)
    ax.set_ylabel('Value', size=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig('Experiments/Charts/Multi/D%V/P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_E&E_d_v_values.png', dpi=300, bbox_inches='tight')
    #plt.show() 
    plt.clf()  
    plt.close('all')  
  
    # Plot d_o_values.
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.7])
    fig.suptitle('P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) +'_E&E_d_o_values', fontsize=12)
    plot_range = [*range(0, iters, 1)]
    
    transposed_d_o_values = np_matrix(d_o_values).T
    for k in range(len(transposed_d_o_values)):
      r = random.uniform(0, 1)
      g = random.uniform(0, 1)
      b = random.uniform(0, 1)
      ax.plot(plot_range, transposed_d_o_values[k].tolist()[0], 
      color=[r, g, b], linewidth=0.5, 
      label='D%O - Dim '+str(k+1))      
    
    ax.axvline(x=convergence_iteration, ymin=0, ymax=1000, 
              label='Convergence', color='black', linestyle='dashed', linewidth=1) # Vertical line showing convergence.
    
    ax.set_xlabel('Iteration', size=10)
    ax.set_ylabel('Value', size=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig('Experiments/Charts/Multi/D%O/P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_E&E_d_o_values.png', dpi=300, bbox_inches='tight')
    #plt.show() 
    plt.clf()  
    plt.close('all') 
  """


  """
  # --------------------
  # E&E Transitions
  # --------------------
  # Get the number of vector in a population.
  pop_len = math.floor(len(all_vectors)/(algorithm_iterations))
  # Calculate 'p' vectors. A 'p' vector represents the presence quantity of each digit in a population at a specific iteration. 
  # ex : (0,0,0,2,0,1,0,0,0,0,0,0,3,0,0,1,1)
  p_vectors = []
  span = 0
  for k in range(int(len(all_vectors)/pop_len)):
    # Get the vectors concerned (belonging to the same population) in order to reconstruct the population of this specific iteration.
    pop = []
    for v in range(pop_len):
      pop.append(all_vectors[span])
      span += 1
    p_vector = sum(pop, axis=0)
    p_vectors.append(p_vector)
  # Calculate 'varp' vectors. A 'varp' vector is the subtract of a 'p' vector by another from a previous iteration. It gives the digits presence variation between two iterations.
  # ex : (0,-2,0,+3,0,0,0,+1,0,-2,-1,0,-1,0,-1,+1,0). 
  # This vector represents, in a way, the evolution of E&E between two iterations. We can also say that this vector carries the transition markers of each digit and therefore that it carries a representation of the transition between E&E, this transition can very well be almost zero or even completely zero. More simply, the varp vector is the marker of a specific transition knowing that this transition can very well appear several times throughout an evolutionary course. It should also be noted that all possible transitions can therefore be represented in the form of a varp vector.
  # It is possible to analyze the evolution of varp vectors over the iterations to see how the research process behaves in terms of E&E.
  varp_vectors = []
  for k in range(1, len(p_vectors)):
    varp = p_vectors[k] - p_vectors[k-1]
    varp_vectors.append(varp)
  # Calculate digits summed presences along iterations. We basically sum each p_vector with all previous p_vectors.
  p_sums = []
  p_sums.append(p_vectors[0])
  for k in range(1, len(p_vectors)):
    p_sum = p_vectors[k] + p_sums[len(p_sums)-1]
    p_sums.append(p_sum)
  # Calculate transition heats. A transition heat is the sum of absolute values of a varp vector. It represents the change quantity between two iterations.
  ths = []
  for k in range(0, len(varp_vectors)):
    th = sum(abs(varp_vectors[k]))
    ths.append(th)
  # Calculate the mean th.
  glob_mean_th = mean(ths)
  # Calculate positive transition heats. A positive transition heat is the sum of positive values of a varp vector.
  ths_pos = []
  for k in range(0, len(varp_vectors)):
    th_pos = 0
    for d in range(len(varp_vectors[k])):
      if varp_vectors[k][d] > 0:
        th_pos += varp_vectors[k][d]
    ths_pos.append(th_pos) 
  # Calculate negative transition heats. A negative transition heat is the sum of negative values of a varp vector.
  ths_neg = []
  for k in range(0, len(varp_vectors)):
    th_neg = 0
    for d in range(len(varp_vectors[k])):
      if varp_vectors[k][d] < 0:
        th_neg += abs(varp_vectors[k][d])
    ths_neg.append(th_neg)

  # Calculate digits summed presences absolute variations along iterations. We basically sum each varp_vector with all previous varp_vectors.
  varp_sums = []
  varp_sums.append(abs(varp_vectors[0]))
  for k in range(1, len(varp_vectors)):
    varp_sum = abs(varp_vectors[k]) + varp_sums[len(varp_sums)-1]
    varp_sums.append(varp_sum)
  """
  
  """ STAY COMMENTED
  # Classify - Using Inner Values
  # Classify varp vectors by calculating their percentage of similarity with each other, regarding their inner values.
  classify_type = "INNER"
  min_similarity_percent = 0.9
  msp_spread = 0.01
  min_class = 3
  max_class = 15
  classes = []
  varp_vectors_classified = []
  while len(classes) < min_class or len(classes) > max_class:
    # Adjust min_similarity_percent
    if len(classes) < min_class:
      min_similarity_percent += msp_spread
    elif len(classes) > max_class:
      min_similarity_percent -= msp_spread
    if min_similarity_percent >= 1 or min_similarity_percent == 0:
      min_similarity_percent = 0.9
      max_class += 1
    print("Classify - min_sim_% = " + str(round(min_similarity_percent, 3)))
    # Classify
    classes = []
    varp_vectors_classified = []
    for k in range(len(varp_vectors)):
      varp_vectors_classified.append({
                                        'varp':varp_vectors[k],
                                        'class':""
                                    })
    class_number = 1
    for k in range(len(varp_vectors_classified)):
      if varp_vectors_classified[k]['class'] == "":
        varp_vectors_classified[k]['class'] = str(class_number)
        classes.append(class_number)
        for m in range(len(varp_vectors_classified)):
          if varp_vectors_classified[m]['class'] == "":
            # Calculate the percentage of similarity, regarding their inner values
            same_values = 0
            for d in range(len(varp_vectors[0])):
              if varp_vectors_classified[k]['varp'][d] == varp_vectors_classified[m]['varp'][d]:
                same_values += 1
            if (same_values/len(varp_vectors[0])) >= min_similarity_percent:
              varp_vectors_classified[m]['class'] = str(class_number)
        class_number += 1
  """
  
  """
  # Classify - Using THs
  # Classify varp vectors by calculating their percentage of similarity with each other, using their THs.
  classify_type = "TH"
  dif_percent_max = 0.2
  dpm_spread = 0.01
  min_class = 3
  max_class = 15
  classes = []
  varp_vectors_classified = []
  while len(classes) < min_class or len(classes) > max_class:
    # Adjust dif_percent_max
    if len(classes) < min_class:
      dif_percent_max -= dpm_spread
    elif len(classes) > max_class:
      dif_percent_max += dpm_spread
    if dif_percent_max >= 1 or dif_percent_max <= 0:
      dif_percent_max = 0.2
      max_class += 1
    #print("Classify - dif_%_max = " + str(round(dif_percent_max, 3)))
    # Classify
    classes = []
    varp_vectors_classified = []
    for k in range(len(varp_vectors)):
      varp_vectors_classified.append({
                                        'varp':varp_vectors[k],
                                        'th':ths[k],
                                        'class':""
                                    })
    class_number = 1
    for k in range(len(varp_vectors_classified)):
      if varp_vectors_classified[k]['class'] == "":
        varp_vectors_classified[k]['class'] = str(class_number)
        classes.append(class_number)
        percent_1 = varp_vectors_classified[k]['th']/glob_mean_th
        for m in range(len(varp_vectors_classified)):
          if varp_vectors_classified[m]['class'] == "":
            # Calculate the percentage of similarity, using their THs.
            # 1/ For both varp, calculate how many percent its th represents towards the glob_mean_th.
            # 2/ Calculate the difference between the two percents obtained.
            # 3/ If the difference represents less than dif_percent_max then the two varp belong to the same class.
            percent_2 = varp_vectors_classified[m]['th']/glob_mean_th
            dif_percent = abs(percent_1 - percent_2)
            if dif_percent < dif_percent_max:
              varp_vectors_classified[m]['class'] = str(class_number)
        class_number += 1
  """




  """
  # Put the transitions class sequence in an array.
  class_sequence = []
  for k in range(len(varp_vectors_classified)):
    class_sequence.append(varp_vectors_classified[k]['class'])
  
  # Classes statistics.
  number_of_classes = max(classes)
  classes_stats = []
  for k in range(1, number_of_classes+1):
    classes_stats.append({
                            'id':k,
                            'varps':[],
                            'mean_varp':[],
                            'number_of_varps':0,
                            'mean_th':0,
                            'mean_th_pos':0,
                            'mean_th_neg':0,
                            'mean_iterations_between_two_varps':0
                        })
    # Varps and Number of varps
    count = 0
    for c in range(len(class_sequence)):
      if class_sequence[c] == str(k):
        count += 1
        classes_stats[-1]['varps'].append(varp_vectors[c])
    classes_stats[-1]['number_of_varps'] = count
    # Mean Varp
    mean_varp = np_mean(classes_stats[-1]['varps'], axis=0)
    classes_stats[-1]['mean_varp'] = mean_varp
    # Mean TH
    mean_th = 0
    for v in range(len(classes_stats[-1]['varps'])):
      mean_th += sum(abs(classes_stats[-1]['varps'][v]))
    mean_th = mean_th / len(classes_stats[-1]['varps'])
    classes_stats[-1]['mean_th'] = int(round(mean_th, 0))
    # Mean TH pos
    mean_th_pos = 0
    for v in range(len(classes_stats[-1]['varps'])):
      for d in range(len(classes_stats[-1]['varps'][v])):  
        if classes_stats[-1]['varps'][v][d] > 0:
          mean_th_pos += classes_stats[-1]['varps'][v][d]
    mean_th_pos = mean_th_pos / len(classes_stats[-1]['varps'])
    classes_stats[-1]['mean_th_pos'] = int(round(mean_th_pos, 0))    
    # Mean TH neg
    mean_th_neg = 0
    for v in range(len(classes_stats[-1]['varps'])):
      for d in range(len(classes_stats[-1]['varps'][v])):  
        if classes_stats[-1]['varps'][v][d] < 0:
          mean_th_neg += abs(classes_stats[-1]['varps'][v][d])
    mean_th_neg = mean_th_neg / len(classes_stats[-1]['varps'])
    classes_stats[-1]['mean_th_neg'] = int(round(mean_th_neg, 0))
    # Mean Iterations Qty Between Varps
    gaps = []
    gap = 0
    for c in range(len(class_sequence)):
      gap += 1
      if class_sequence[c] == str(k):
        gaps.append(gap)
        gap = 0
    mean_gap = 0 # Default, if there is only one varp in the class
    if len(gaps) > 1:
      mean_gap = mean(gaps)
    classes_stats[-1]['mean_iterations_between_two_varps'] = mean_gap
  """


  """
  # Plot the mean_varp of each class.
  mean_varps = []
  for k in range(len(classes_stats)):
    mean_varps.append(classes_stats[k]['mean_varp'])
  fig = plt.figure(figsize=(8,5))
  ax = fig.add_axes([0.1, 0.1, 0.6, 0.7])
  fig.suptitle('P' + str(profile_number) + ' ' + current_algo + ' ' + constraints_variation_name + ' Stag' + str(stagnation) + ' Classify-' + classify_type +' E&E_Mean_Varp_Of_Each_Class', fontsize=12)
  plot_range = [*range(1, len(all_vectors[0])+1, 1)]
  for k in range(len(mean_varps)):
    r = random.uniform(0, 1)
    g = random.uniform(0, 1)
    b = random.uniform(0, 1)
    ax.plot(plot_range, mean_varps[k], color=[r, g, b], linewidth=0.5, label='Mean Varp - Class '+str(k+1))
  ax.set_xlabel('Digit', size=10)
  ax.set_ylabel('Mean Variation Value', size=10)
  ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.savefig('Experiments/Charts/Transition/P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_Classify-' + classify_type + '_Mean_Varp_Of_Each_Class.png', dpi=300, bbox_inches='tight')
  #plt.show() 
  plt.clf()
  # Close all figures.
  plt.close('all')

  # Plot the mean number of iterations separating two varps in each class.
  gaps = []
  for k in range(len(classes_stats)):
    gaps.append(round(classes_stats[k]['mean_iterations_between_two_varps'],1))  
  # Plot
  classes_names = []
  for k in range(len(classes_stats)): 
    classes_names.append(classes_stats[k]['id'])
  fig = plt.figure(figsize=(8,5))
  ax = fig.add_axes([0.1, 0.1, 0.6, 0.7])
  fig.suptitle('P' + str(profile_number) + ' ' + current_algo + ' ' + constraints_variation_name + ' Stag' + str(stagnation) + ' Classify-' + classify_type + ' E&E_Transitions_Class - Mean Iterations Qty Between Varps', fontsize=12)
  plt.title('NOTA : Null value = only one varp in the class', fontsize=10)
  values = gaps
  ax.bar(classes_names,values)     
  ax.set_xlabel('Transition Class', size=10)
  ax.set_ylabel('Mean Iterations Qty Between Varps', size=10)
  for i in range(0, len(classes_names)): # Display values on bars
    plt.text(i+1,values[i]+0.1,values[i], ha = 'center')
  ax.set_xticks(classes_names)
  plt.savefig('Experiments/Charts/Transition/P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_Classify-' + classify_type + '_E&E_Transitions_Class_Mean_Iterations_Qty_Between_Varps.png', dpi=300, bbox_inches='tight')
  #plt.show() 
  plt.clf()
  # Close all figures.
  plt.close('all')
  
  # Plot mean_th, mean_th_pos and mean_th_neg of each class.
  mean_ths = []
  for k in range(len(classes_stats)):
    mean_ths.append(classes_stats[k]['mean_th'])
  mean_ths_pos = []
  for k in range(len(classes_stats)):
    mean_ths_pos.append(classes_stats[k]['mean_th_pos'])
  mean_ths_neg = []
  for k in range(len(classes_stats)):
    mean_ths_neg.append(classes_stats[k]['mean_th_neg'])    
  # Plots
  classes_names = []
  for k in range(len(classes_stats)): 
    classes_names.append(classes_stats[k]['id'])
  x = arange(len(classes_names))
  fig = plt.figure(figsize=(8,5))
  ax = fig.add_axes([0.1, 0.1, 0.6, 0.7])
  fig.suptitle('P' + str(profile_number) + ' ' + current_algo + ' ' + constraints_variation_name + ' Stag' + str(stagnation) + ' Classify-' + classify_type + ' E&E_Transitions_Class - Mean_THs', fontsize=12)
  mean_ths_values = mean_ths
  mean_ths_pos_values = mean_ths_pos
  mean_ths_neg_values = mean_ths_neg
  width = 0.2
  bar_1 = ax.bar(x-0.2,mean_ths_neg_values, width, label='Mean TH Neg', color='orange') 
  bar_2 = ax.bar(x,mean_ths_values, width, label='Mean TH', color='b')  
  bar_3 = ax.bar(x+0.2,mean_ths_pos_values, width, label='Mean TH Pos', color='g')   
  ax.set_xlabel('Transition Class', size=10)
  ax.set_ylabel('Mean TH', size=10)
  for i in range(0, len(classes_names)): # Display values on bars
    plt.text(i-0.2,mean_ths_neg_values[i]+0.3,mean_ths_neg_values[i], ha='center')
    plt.text(i,mean_ths_values[i]+0.3,mean_ths_values[i], ha='center')
    plt.text(i+0.2,mean_ths_pos_values[i]+0.3,mean_ths_pos_values[i], ha='center')
  ax.set_xticks(x)
  ax.set_xticklabels(classes_names)
  ax.legend()
  plt.savefig('Experiments/Charts/Transition/P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_Classify-' + classify_type + '_E&E_Transitions_Class_Mean_THs.png', dpi=300, bbox_inches='tight')
  #plt.show() 
  plt.clf()
  # Close all figures.
  plt.close('all')
  
  # Plot the number of varps in each class.
  varps_quantities = []
  for k in range(len(classes_stats)):
    varps_quantities.append(classes_stats[k]['number_of_varps'])
  # Plot
  classes_names = []
  for k in range(len(classes_stats)): 
    classes_names.append(classes_stats[k]['id'])
  fig = plt.figure(figsize=(8,5))
  ax = fig.add_axes([0.1, 0.1, 0.6, 0.7])
  fig.suptitle('P' + str(profile_number) + ' ' + current_algo + ' ' + constraints_variation_name + ' Stag' + str(stagnation) + ' Classify-' + classify_type + ' E&E_Transitions_Class - Varps_Quantities', fontsize=12)
  values = varps_quantities
  ax.bar(classes_names,values)     
  ax.set_xlabel('Transition Class', size=10)
  ax.set_ylabel('Varps Quantity', size=10)
  for i in range(0, len(classes_names)): # Display values on bars
    plt.text(i+1,values[i],values[i], ha = 'center')
  ax.set_xticks(classes_names)
  plt.savefig('Experiments/Charts/Transition/P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_Classify-' + classify_type + '_E&E_Transitions_Class_Varps_Quantities.png', dpi=300, bbox_inches='tight')
  #plt.show() 
  plt.clf()
  # Close all figures.
  plt.close('all')
  
  # Plot the transitions class sequence.
  fig = plt.figure(figsize=(8,5))
  ax = fig.add_axes([0.1, 0.1, 0.6, 0.7])
  fig.suptitle('P' + str(profile_number) + ' ' + current_algo + ' ' + constraints_variation_name + ' Stag' + str(stagnation) + ' Classify-' + classify_type + ' E&E_Transitions_Class - Sequence', fontsize=12)
  plot_range = [*range(0, len(class_sequence), 1)]
  ax.plot(plot_range, class_sequence, 'b.', linewidth=0.5)      
  ax.axvline(x=convergence_iteration, ymin=0, ymax=1000, 
            label='Convergence', color='black', linestyle='dashed', linewidth=1) # Vertical line showing convergence.
  
  the_label = 'Exploration'
  for k in range(len(ee_phases)):
    if ee_phases[k] == "Exploration":
      ax.axvline(x=k, ymin=0, ymax=1000, 
              label=the_label, color='black', linestyle='dashed', linewidth=0.4)
      the_label = ""
  
  ax.set_xlabel('Iteration', size=10)
  ax.set_ylabel('Transition Class', size=10)
  ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.savefig('Experiments/Charts/Transition/P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_Classify-' + classify_type + '_E&E_Transitions_Class_Sequence.png', dpi=300, bbox_inches='tight')
  
  plt.savefig('Experiments/Charts/ee_phases based on classification/P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_Classify-' + classify_type + '_E&E_Transitions_Class_Sequence.png', dpi=300, bbox_inches='tight')
  
  #plt.show() 
  plt.clf()
  # Close all figures.
  plt.close('all')
  
  # Plot Transition Heats.
  fig = plt.figure(figsize=(8,5))
  ax = fig.add_axes([0.1, 0.1, 0.6, 0.7])
  fig.suptitle('P' + str(profile_number) + ' ' + current_algo + ' ' + constraints_variation_name + ' Stag' + str(stagnation) + ' E&E_Transition_Heats', fontsize=12)
  plot_range = [*range(0, len(ths), 1)]
  ax.plot(plot_range, ths, 'b-', linewidth=0.5, 
            label='THS')      
  ax.plot(plot_range, ths_pos, 'g-', linewidth=0.5, 
            label='THS POS')
  ax.plot(plot_range, ths_neg, 'r-', linewidth=0.5, 
            label='THS NEG')
  ax.axvline(x=convergence_iteration, ymin=0, ymax=1000, 
            label='Convergence', color='black', linestyle='dashed', linewidth=1) # Vertical line showing convergence.
  
  the_label = 'Exploration'
  for k in range(len(ee_phases)):
    if ee_phases[k] == "Exploration":
      ax.axvline(x=k, ymin=0, ymax=1000, 
              label=the_label, color='black', linestyle='dashed', linewidth=0.4)
      the_label = ""
  
  ax.set_xlabel('Iteration', size=10)
  ax.set_ylabel('TH', size=10)
  ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.savefig('Experiments/Charts/Transition/P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_E&E_Transition_Heats.png', dpi=300, bbox_inches='tight')
  
  #plt.show() 
  plt.clf()
  # Close all figures.
  plt.close('all')
  
  # Plot digits summed presence absolute variations evolution.
  fig = plt.figure(figsize=(8,5))
  ax = fig.add_axes([0.1, 0.1, 0.6, 0.7])
  fig.suptitle('P' + str(profile_number) + ' ' + current_algo + ' ' + constraints_variation_name + ' Stag' + str(stagnation) +' E&E_Digits_Absolute_Summed_Presence_Variations', fontsize=12)
  plot_range = [*range(0, len(varp_sums), 1)]
  transposed_varp_sums = np_matrix(varp_sums).T
  for k in range(len(transposed_varp_sums)):
    r = random.uniform(0, 1)
    g = random.uniform(0, 1)
    b = random.uniform(0, 1)
    ax.plot(plot_range, transposed_varp_sums[k].tolist()[0], 
    color=[r, g, b], linewidth=0.5)      
  ax.axvline(x=convergence_iteration, ymin=0, ymax=1000, 
            label='Convergence', color='black', linestyle='dashed', linewidth=1) # Vertical line showing convergence.
  
  the_label = 'Exploration'
  for k in range(len(ee_phases)):
    if ee_phases[k] == "Exploration":
      ax.axvline(x=k, ymin=0, ymax=1000, 
              label=the_label, color='black', linestyle='dashed', linewidth=0.4)
      the_label = ""
  
  ax.set_xlabel('Iteration', size=10)
  ax.set_ylabel('Absolute Summed Presence Variation', size=10)
  ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.savefig('Experiments/Charts/Transition/P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_Digits_Presence_Variation_Summed.png', dpi=300, bbox_inches='tight')
  
  plt.savefig('Experiments/Charts/ee_phases based on classification/P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_Digits_Presence_Variation_Summed.png', dpi=300, bbox_inches='tight')
  
  #plt.show() 
  plt.clf()
  # Close all figures.
  plt.close('all')  
  
  # Plot digits summed presence evolution.
  fig = plt.figure(figsize=(8,5))
  ax = fig.add_axes([0.1, 0.1, 0.6, 0.7])
  fig.suptitle('P' + str(profile_number) + ' ' + current_algo + ' ' + constraints_variation_name + ' Stag' + str(stagnation) +' E&E_Digits_Summed_Presence', fontsize=12)
  plot_range = [*range(0, len(p_sums), 1)]
  transposed_p_sums = np_matrix(p_sums).T
  for k in range(len(transposed_p_sums)):
    r = random.uniform(0, 1)
    g = random.uniform(0, 1)
    b = random.uniform(0, 1)
    ax.plot(plot_range, transposed_p_sums[k].tolist()[0], 
    color=[r, g, b], linewidth=0.5)      
  ax.axvline(x=convergence_iteration, ymin=0, ymax=1000, 
            label='Convergence', color='black', linestyle='dashed', linewidth=1) # Vertical line showing convergence.
  
  the_label = 'Exploration'
  for k in range(len(ee_phases)):
    if ee_phases[k] == "Exploration":
      ax.axvline(x=k, ymin=0, ymax=1000, 
              label=the_label, color='black', linestyle='dashed', linewidth=0.4)
      the_label = ""  
  
  ax.set_xlabel('Iteration', size=10)
  ax.set_ylabel('Summed Presence', size=10)
  ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.savefig('Experiments/Charts/Transition/P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_Digits_Summed_Presence.png', dpi=300, bbox_inches='tight')
  
  plt.savefig('Experiments/Charts/ee_phases based on classification/P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_Digits_Summed_Presence.png', dpi=300, bbox_inches='tight')
  
  #plt.show() 
  plt.clf()
  # Close all figures.
  plt.close('all')
  """
  
  # Plot digits presence evolution. => Unreadable
  """ STAY COMMENTED
  fig = plt.figure(figsize=(8,5))
  ax = fig.add_axes([0.1, 0.1, 0.6, 0.7])
  fig.suptitle('P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) +'_E&E_Digits_Presence', fontsize=12)
  plot_range = [*range(0, algorithm_iterations+1, 1)]
  transposed_p_vectors = np_matrix(p_vectors).T
  for k in range(len(transposed_p_vectors)):
    r = random.uniform(0, 1)
    g = random.uniform(0, 1)
    b = random.uniform(0, 1)
    ax.plot(plot_range, transposed_p_vectors[k].tolist()[0], 
    color=[r, g, b], linewidth=0.5, 
    label='Digit Presence'+str(k+1))      
  ax.axvline(x=convergence_iteration, ymin=0, ymax=1000, 
            label='Convergence', color='black', linestyle='dashed', linewidth=1) # Vertical line showing convergence.
  ax.set_xlabel('Iteration', size=10)
  ax.set_ylabel('Presence', size=10)
  ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.savefig('Experiments/Charts/Transition/P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_Digits_Presence.png', dpi=300, bbox_inches='tight')
  #plt.show() 
  plt.clf()      
  """

  # Plot digits presence variations evolution. => Unreadable
  """ STAY COMMENTED
  fig = plt.figure(figsize=(8,5))
  ax = fig.add_axes([0.1, 0.1, 0.6, 0.7])
  fig.suptitle('P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) +'_E&E_Digits_Presence_Variation', fontsize=12)
  plot_range = [*range(0, algorithm_iterations, 1)]
  transposed_varp_vectors = np_matrix(varp_vectors).T
  for k in range(len(transposed_varp_vectors)):
    r = random.uniform(0, 1)
    g = random.uniform(0, 1)
    b = random.uniform(0, 1)
    ax.plot(plot_range, transposed_varp_vectors[k].tolist()[0], 
    color=[r, g, b], linewidth=0.5, 
    label='Digit Presence Variation'+str(k+1))      
  ax.axvline(x=convergence_iteration, ymin=0, ymax=1000, 
            label='Convergence', color='black', linestyle='dashed', linewidth=1) # Vertical line showing convergence.
  ax.set_xlabel('Iteration', size=10)
  ax.set_ylabel('Presence Variation', size=10)
  ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.savefig('Experiments/Charts/Transition/P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_Digits_Presence_Variations.png', dpi=300, bbox_inches='tight')
  #plt.show() 
  plt.clf()   
  """
  
  plt.close('all')
  # ----------------------------------------
  # End of "E&E Transitions" section
  # ----------------------------------------



  # ------------------------------------------
  # Display E&E Phases Based On Classification
  # ------------------------------------------
  # See function add_to_vectors_classes on file functions.py for more details.
  # The goal of this visualisation is to represent the alternation of E&E phases through iterations using ee_phases which was fed using a classification of vectors through iterations.
  # The goal is also to look for correlations with some of the evaluation criteria (score, etc). If interesting correlations can be observed, it would validate the relevance of this approach regarding the representation of E&E.
  
  # NOTA : for criteria "Summed Presences", "Summed Presences Variations" and "Classes Interventions", we added the vertical exploration lines to corresponding charts on previous section called "E&E Transitions".
  
  """
  # Display Multi
  fig = plt.figure(figsize=(8,5))
  ax = fig.add_axes([0.1, 0.1, 0.6, 0.7])
  fig.suptitle('P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_E&E_Phases_Based_On_Classification_Multi', fontsize=12)
  plt.title('E&E_Phases_Based_On_Classification_Multi', fontsize=10)
  plot_range = [*range(0, iters, 1)]
  
  ax.plot(plot_range, powers, 'b-', linewidth=0.5, 
            label='Power')
  ax.plot(plot_range, top_vectors_scores, 'c-', linewidth=0.5, 
            label='TopV Score (scaled)')
  ax.plot(plot_range, all_vectors_coverages, 'm-', linewidth=0.5, 
            label='AllV Coverage')
  ax.plot(plot_range, top_vectors_constraints_completions, 'y-', 
            linewidth=0.5, 
            label='TopV Const Comp')
  ax.plot(plot_range, similarities_with_top_v, color='#FF9E00', 
            linewidth=0.5, 
            label='Similarity with TopV')          
            
  ax.plot(plot_range, hvv_values, 'r-', linewidth=0.5, 
            label='HVV')      
  ax.plot(plot_range, hvo_values, 'r--', linewidth=0.5, 
            label='HVO')
  ax.plot(plot_range, hvcv_values, 'g-', linewidth=0.5, 
            label='HVCV (scaled)')
  ax.plot(plot_range, hvco_values, 'g--', linewidth=0.5, 
            label='HVCO (scaled)') 
              
  ax.axvline(x=convergence_iteration, ymin=0, ymax=1000, label='Convergence', color='black', linestyle='dashed', linewidth=1)
  
  the_label = 'Exploration'
  for k in range(len(ee_phases)):
    if ee_phases[k] == "Exploration":
      ax.axvline(x=k, ymin=0, ymax=1000, 
              label=the_label, color='black', linestyle='dashed', linewidth=0.4)
      the_label = ""
  
  ax.set_xlabel('Iteration', size=10)
  ax.set_ylabel('Value', size=10)
  ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.savefig('Experiments/Charts/ee_phases based on classification/P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_E&E_Phases_Based_On_Classification - Multi.png', dpi=300, bbox_inches='tight')
  #plt.show() 
  plt.clf()
  # Close all figures.
  plt.close('all')
  
  
  # Display iteration_score_evolution and max_score_evolution
  fig = plt.figure(figsize=(8,5))
  ax = fig.add_axes([0.1, 0.1, 0.6, 0.7])
  fig.suptitle('P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_E&E_Phases_Based_On_Classification_Scores', fontsize=12)
  plt.title('E&E_Phases_Based_On_Classification_Scores', fontsize=10)
  plot_range = [*range(0, algorithm_iterations, 1)]
  
  ax.plot(plot_range, iteration_score_evolution, 'g-', linewidth=0.5, markersize=1, label='Iteration Score Evolution')
            
  ax.plot(plot_range, max_score_evolution, 'r-', linewidth=0.5, markersize=1, label='Max Score Evolution')
  
  ax.axvline(x=convergence_iteration, ymin=0, ymax=1000, label='Convergence', color='black', linestyle='dashed', linewidth=1)
  
  the_label = 'Exploration'
  for k in range(len(ee_phases)):
    if ee_phases[k] == "Exploration":
      ax.axvline(x=k, ymin=0, ymax=1000, 
              label=the_label, color='black', linestyle='dashed', linewidth=0.4)
      the_label = ""
  
  ax.set_xlabel('Iteration', size=10)
  ax.set_ylabel('Value', size=10)
  ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.savefig('Experiments/Charts/ee_phases based on classification/P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_E&E_Phases_Based_On_Classification - Scores.png', dpi=300, bbox_inches='tight')
  #plt.show() 
  plt.clf()
  # Close all figures.
  plt.close('all')
  """
  
  # ------------------------------------------
  # End of :
  # Display E&E Phases Based On Classification
  # ------------------------------------------




  
  """
  # Display final recommendations information.
  for k in range(0, len(final_vectors_selection)):
    # Display Actor and total resources recommended.
    total_resources_recommended = sum(final_vectors_selection[k]['vector'])
    print("-----------------------------------")
    print("Actor " + str(k+1))
    print("Total Resources Recommended : " + str(total_resources_recommended))
    
    # Display detailed recommended resources.
    for i in range(len(final_vectors_selection[k]['vector'])):
      if final_vectors_selection[k]['vector'][i] == 1:
        print("Resource " + df[i][0] + " is recommended.")
  print("-----------------------------------")
  """
  
  # Display mean score.
  print("Mean Score = " + str(round(final_mean_score, 2)))
  
  # Display similarity percentage.
  #print("Mean Similarity = " + str(int(100*final_mean_similarity)) + "%")
  
  # Display popularity.
  #print("Popularity = " + str(int(100*popularity)) + "%")
  
  # Display global constraints completion.
  #print("Constraints Completion = " + str(int(100*global_constraints_completion)) + "%")
  
  # Display links gap.
  #print("Links Gap = " + str(int(100*links_gap)) + "%")
  
  # Display vector links gap.
  #print("Vector Links Gap = " + str(int(100*vector_links_gap)) + "%")
  
  print("-----------------------------------")
  
  
  return [
            #final_mean_similarity, 
            #final_mean_score, 
            dense_vector, 
            presence_gaps, 
            presence_rate, 
            all_vectors_coverage, 
            top_vectors_coverage, 
            popularity,
            global_constraints_completion,
            mean_potential,
            links_gap,
            vector_links_gap,
            ee_power,
            #counter_explore_iter,
            #counter_exploit_iter,
            #counter_explore_vector,
            #counter_exploit_vector,
            #counter_explore_digit,
            #counter_exploit_digit,
            min_eit_gap,
            max_eit_gap,
            diff_min_max_eit_gaps,
            mean_eit_gap,
            mean_diff_between_eit_gaps
          ]
 
  
  
  
# Guide Exploration & Exploitation (E&E).
# Function to check if the balance between E&E is good regarding thresholds on a global indicator. If the balance is good, the function returns the population without modifications, else the function returns the population modified. 
# Before being returned, the population is normalised.
# The population parameter must contain at least two vectors.
def guide_explo_with_merged_EE_cases(population):

  # Number of vectors in the population.
  pop_size = len(population)  
  
  # Get a vector representing how many times each digit is a 1-digit in the entire population.
  summed_vectors = sum(population, axis=0)
  
  # Calculate the global indicator of similarity between population vectors.
  # To do that, we divide each digit's number of appearance by the number of vectors in the population. At the end, we do a mean of all the values obtained.
  # The maximum value of e_e_indicator is 1. So, e_e_indicator is a percentage. This percentage represents the mean number of presence, of each digit, in all the population vectors. So you can read this percentage as "Each digit is present in x% of vectors" or as "Vectors' similarity is x%".
  # The more e_e_indicator is high, the more the vectors are similar corresponding to a behaviour too much centered on exploitation.
  # The more e_e_indicator is low, the more the vectors are different corresponding to a behaviour too much centered on exploration. 
  digit_values = []
  for k in range(len(summed_vectors)):
    if summed_vectors[k] > 0:
      digit_values.append(summed_vectors[k]/pop_size) 
  e_e_indicator = mean(digit_values)

  # Save e_e_indicator values in order to adjust e_e_min and e_e_max.
  e_e_indicator_values = []
  e_e_indicator_values.append(e_e_indicator)
  
  # Starting limits min and max for e_e_indicator + tolerance spread around the mean e_e_indicator.
  e_e_min = 0.5
  e_e_max = 0.8
  spread = 0.05
  
  #print("START GUIDING")

  # Force E&E while limits are not respected.
  while e_e_indicator <= e_e_min or e_e_indicator >= e_e_max:
    
    #print("===> FORCE E&E - Indicator= " + str(round(e_e_indicator, 2)) + " ", end="\r")
    #print("===> FORCE E&E - Indicator= " + str(round(e_e_indicator, 2)))
    
    """
    # Remove 25% of the worst vectors from population.
    #print("==================================> REMOVE 25%")
    remove_quantity = int(0.25 * pop_size)
    if remove_quantity < 1:
      remove_quantity = 1
    # Sort population by score DESC.
    population = sorted(deepcopy(population), key=score, reverse=True)
    # Remove
    for k in range(remove_quantity):
      population.pop()
    # Add in the population as many fresh generated vectors as the quantity removed.
    for k in range(remove_quantity):
      population.append(generate_vector())
    """
    
    # In each vector of population, a 0-digit becomes a 1-digit while constraints are respected.
    for k in range(len(population)):
      done = False
      while done is False:
        rand = random.randint(0, len(population[k])-1)
        if population[k][rand] == 0:
          population[k][rand] = 1
          if constraints_max_values_check(population[k]) is False:
            done = True # End while loop.
            population[k][rand] = 0 # CANCEL LAST MODIFICATION
    
    # Adjust e_e_min and e_e_max using the tolerance spread toward mean e_e_indicator.
    mean_e_e_indicator = mean(e_e_indicator_values)
    e_e_min = mean_e_e_indicator - spread
    if e_e_min < 0: 
      e_e_min = 0
    e_e_max = mean_e_e_indicator + spread
    """
    if e_e_max > 1: 
      e_e_max = 1
    """
    
    # Update e_e_indicator and add it to e_e_indicator_values.
    summed_vectors = sum(population, axis=0)
    digit_values = []
    for k in range(len(summed_vectors)):
      if summed_vectors[k] > 0:
        digit_values.append(summed_vectors[k]/pop_size) 
    e_e_indicator = mean(digit_values)
    e_e_indicator_values.append(e_e_indicator)

  
  # Normalise the population.
  for k in range(len(population)):
    population[k] = normalise(deepcopy(population[k]))
  
  return population




# Guide Exploration & Exploitation (E&E) With Separated Cases
# Function to check if the balance between E&E is good regarding thresholds on a global indicator. If the balance is good, the function returns the population without modifications, else the function returns the population modified according to the E&E situation (force exploration or force exploitation). 
# Before being returned, the population is normalised.
# The population parameter must contain at least two vectors.
def guide_explo_with_separated_EE_cases(population):

  # Number of vectors in the population.
  pop_size = len(population)  
  
  # Get a vector representing how many times each digit is a 1-digit in the entire population.
  summed_vectors = sum(population, axis=0)
  
  # Calculate the global indicator of similarity between population vectors.
  # To do that, we divide each digit's number of appearance by the number of vectors in the population. At the end, we do a mean of all the values obtained.
  # The maximum value of e_e_indicator is 1. So, e_e_indicator is a percentage. This percentage represents the mean number of presence, of each digit, in all the population vectors. So you can read this percentage as "Each digit is present in x% of vectors" or as "Vectors' similarity is x%".
  # The more e_e_indicator is high, the more the vectors are similar corresponding to a behaviour too much centered on exploitation.
  # The more e_e_indicator is low, the more the vectors are different corresponding to a behaviour too much centered on exploration. 
  digit_values = []
  for k in range(len(summed_vectors)):
    if summed_vectors[k] > 0:
      digit_values.append(summed_vectors[k]/pop_size) 
  e_e_indicator = mean(digit_values)

  # Save e_e_indicator values in order to adjust e_e_min and e_e_max.
  e_e_indicator_values = []
  e_e_indicator_values.append(e_e_indicator)
  
  # Starting limits min and max for e_e_indicator + tolerance spread around the mean e_e_indicator.
  e_e_min = 0.5
  e_e_max = 0.8
  spread = 0.05
  
  # Guiding iterations limit used to avoid being trapped into an infinite loop.
  i_limit = 10
  
  #print("START GUIDING")
  i = 0
  while (e_e_indicator <= e_e_min or e_e_indicator >= e_e_max) and i <= i_limit:
    i += 1

    while e_e_indicator <= e_e_min and i <= i_limit:
      i += 1
      #print("===> FORCE EXPLOITATION - Indicator= " + str(round(e_e_indicator, 2)) + " ", end="\r")
      #print("===> FORCE EXPLOITATION - Indicator= " + str(round(e_e_indicator, 2)))
      
      # In each vector of population, a 0-digit becomes a 1-digit while constraints are respected.
      for k in range(len(population)):
        done = False
        while done is False:
          rand = random.randint(0, len(population[k])-1)
          if population[k][rand] == 0:
            population[k][rand] = 1
            if constraints_max_values_check(population[k]) is False:
              done = True # End while loop.
              population[k][rand] = 0 # CANCEL LAST MODIFICATION
      
      # Adjust e_e_min and e_e_max using the tolerance spread toward mean e_e_indicator.
      mean_e_e_indicator = mean(e_e_indicator_values)
      e_e_min = mean_e_e_indicator - spread
      if e_e_min < 0: 
        e_e_min = 0
      e_e_max = mean_e_e_indicator + spread
      """
      if e_e_max > 1: 
        e_e_max = 1
      """
      
      # Update e_e_indicator and add it to e_e_indicator_values.
      summed_vectors = sum(population, axis=0)
      digit_values = []
      for k in range(len(summed_vectors)):
        if summed_vectors[k] > 0:
          digit_values.append(summed_vectors[k]/pop_size) 
      e_e_indicator = mean(digit_values)
      e_e_indicator_values.append(e_e_indicator)


    while e_e_indicator >= e_e_max and i <= i_limit:
      i += 1
      #print("===> FORCE EXPLORATION - Indicator= " + str(round(e_e_indicator, 2)) + " ", end="\r")
      #print("===> FORCE EXPLORATION - Indicator= " + str(round(e_e_indicator, 2)))
      
      # Remove 25% of the worst vectors from population.
      #print("==================================> REMOVE 25%")
      remove_quantity = int(0.25 * pop_size)
      if remove_quantity < 1:
        remove_quantity = 1
      # Sort population by score DESC.
      population = sorted(deepcopy(population), key=score, reverse=True)
      # Remove
      for k in range(remove_quantity):
        population.pop()
      # Add in the population as many fresh generated vectors as the quantity removed.
      for k in range(remove_quantity):
        population.append(generate_vector())
      
      # Adjust e_e_min and e_e_max using the tolerance spread toward mean e_e_indicator.
      mean_e_e_indicator = mean(e_e_indicator_values)
      e_e_min = mean_e_e_indicator - spread
      if e_e_min < 0: 
        e_e_min = 0
      e_e_max = mean_e_e_indicator + spread
      """
      if e_e_max > 1: 
        e_e_max = 1
      """
      
      # Update e_e_indicator and add it to e_e_indicator_values.
      summed_vectors = sum(population, axis=0)
      digit_values = []
      for k in range(len(summed_vectors)):
        if summed_vectors[k] > 0:
          digit_values.append(summed_vectors[k]/pop_size) 
      e_e_indicator = mean(digit_values)
      e_e_indicator_values.append(e_e_indicator)

  
  # Normalise the population.
  for k in range(len(population)):
    population[k] = normalise(deepcopy(population[k]))
  
  return population




# Function to get the best and worse digits from df regarding score and constraints score.
# The return from this function is used as parameter for the function guide_explo_with_potential().
# The quantity parameter allows to get, for each case considered, this precise number of best or worse digits.
def get_best_and_worse_digits(quantity):

  # The digits array that is returned.
  digits = []
  
  # Build an array containing tuples (digit_index, score, constraints_score).
  indexes_scores = []
  for k in range(len(df)):
    indexes_scores.append({
                            'index':k,
                            'score':resource_score(k),
                            'constraints_score':resource_constraints_score(k)
                          })   

  # Find the best and worse digit indexes regarding score and constraints score.
  # ---------------------------------------------------------------------------
  # DEALING WITH THE SCORE
  # Sort indexes_scores by score DESC.  
  indexes_scores = sorted(deepcopy(indexes_scores), key=lambda dct: dct['score'], reverse=True)
  # 1st CASE
  # The best digits, regarding score have to be 1-digits.
  for k in range(0, quantity):
    # Add to digits
    digits.append({
                    'index':indexes_scores[k]['index'],
                    'wanted_value':1
                  })
  # 2nd CASE
  # The worse digits, regarding score, have to be 0-digits.
  for k in range(len(indexes_scores)-quantity, len(indexes_scores)):
    # Add to digits
    digits.append({
                    'index':indexes_scores[k]['index'],
                    'wanted_value':0
                  })  
  # ---------------------------------------------------------------------------
  # DEALING WITH THE CONSTRAINTS SCORE
  # Sort indexes_scores by constraints_score DESC.
  indexes_scores = sorted(deepcopy(indexes_scores), key=lambda dct: dct['constraints_score'], reverse=True)
  # 1st CASE
  # The best digits, regarding constraints score have to be 0-digits.
  for k in range(0, quantity):
    # Add to digits
    digits.append({
                    'index':indexes_scores[k]['index'],
                    'wanted_value':0
                  })
  # 2nd CASE
  # The worse digits, regarding constraints score, have to be 1-digits.
  for k in range(len(indexes_scores)-quantity, len(indexes_scores)):
    # Add to digits
    digits.append({
                    'index':indexes_scores[k]['index'],
                    'wanted_value':1
                  })
                  
  return digits




# Guide Exploration & Exploitation (E&E) using a Potential indicator.
# Function to check the potential of improvment of the population, regarding obvious improvment cases inside each vector.
# If the future potential of the population is above a threshold max, we consider that its situation toward E&E is not good and we apply variations to it.
# Before being returned, the population is normalised.
# The population parameter must contain at least two vectors.
# best_worse_digits is returned by calling the get_best_and_worse_digits(quantity) function (see main.py).
def guide_explo_with_potential(population):
  
  # Potential max limit to know if we have to do variations on the population + tolerance spread around the mean potential.
  potential_max = 0.3
  spread = 0.05

  # For each vector in the population, calculate its potential.
  potentials = []
  for p in range(len(population)):
    potential = 0
    for d in range(len(best_worse_digits)):  
      # If the vector value is not the wanted value, we increment its potential.
      if population[p][best_worse_digits[d]['index']] != best_worse_digits[d]['wanted_value']:
        potential += 1
    # Final potential as a percentage.
    potential /= len(best_worse_digits)
    # Append to potentials
    potentials.append(potential)
  
  # Calculate the mean potential of the population.
  pop_pot = mean(potentials)

  # Append pop_pot to potential_values in order to adjust potential_max later on.
  potential_values = []
  potential_values.append(pop_pot)
  
  #print("START GUIDING")
    
  # Apply variations to the population while its potential is too high.
  while pop_pot >= potential_max:
    
    #print("FORCE EXPLO - Potential=" + str(round(potential,2)))
    
    """
    # Remove 25% of the worst vectors from population.
    #print("==================================> REMOVE 25%")
    remove_quantity = int(0.25 * len(population))
    if remove_quantity < 1:
      remove_quantity = 1
    # Sort population by score DESC.
    population = sorted(deepcopy(population), key=score, reverse=True)
    # Remove
    for k in range(remove_quantity):
      population.pop()
    # Add in the population as many fresh generated vectors as the quantity removed.
    for k in range(remove_quantity):
      population.append(generate_vector())
    """
    for p in range(len(population)):
      # A 0-digit becomes a 1-digit while constraints are respected.
      done = False
      while done is False:
        rand = random.randint(0, len(population[p])-1)
        if population[p][rand] == 0:
          population[p][rand] = 1
          if constraints_max_values_check(population[p]) is False:
            done = True # End while loop.
            population[p][rand] = 0 # CANCEL LAST MODIFICATION    

    # Adjust potential_max using the tolerance spread around the mean potential.
    mean_potential = mean(potential_values)
    potential_max = mean_potential + spread
    if potential_max > 1: 
      potential_max = 1

    # Update pop_pot and add it to potential_values.
    # For each vector in the population, calculate its potential.
    potentials = []
    for p in range(len(population)):
      potential = 0
      for d in range(len(best_worse_digits)):  
        # If the vector value is not the wanted value, we increment its potential.
        if population[p][best_worse_digits[d]['index']] != best_worse_digits[d]['wanted_value']:
          potential += 1
      # Final potential as a percentage.
      potential /= len(best_worse_digits)
      # Append to potentials
      potentials.append(potential)
    # Calculate the mean potential of the population.
    pop_pot = mean(potentials)
    # Append to potential_values
    potential_values.append(pop_pot)


  # Normalise the population.
  for k in range(len(population)):
    population[k] = normalise(deepcopy(population[k]))
  
  return population




# Function to get the constraints completion scores of each digit in df.
# The return from this function is used as parameter for the function guide_explo_with_cc().
def get_digits_cc_scores():

  digits_cc_scores = []
  
  for d in range(len(df)):
    digits_cc_scores.append(resource_constraints_score(d))
  
  return digits_cc_scores




# Guide Exploration & Exploitation (E&E) using an indicator based on the constraints completion (volume pressure) in the population.
# If this indicator is below a threshold min, we consider that the situation toward E&E is not good and we apply variations to the population.
# Before being returned, the population is normalised.
# The population parameter must contain at least two vectors.
# digits_cc_scores is returned by calling the get_digits_cc_scores() function (see main.py).
def guide_explo_with_cc(population):

  # Minimum threshold for the cc_indicator + spread around the mean cc_indicator.
  cc_min = 0.9
  spread = 0.05
  
  # Calculate the constraints completion in the population.
  population_constraints_completion = 0
  vectors_constraints_completion = []
  for k in range(len(population)):
    digits_constraints_completion = []
    for d in range(len(population[k])):
      if population[k][d] == 1:
        digits_constraints_completion.append(digits_cc_scores[d])
    vectors_constraints_completion.append(sum(digits_constraints_completion))
  population_constraints_completion = mean(vectors_constraints_completion)
  
  # The cc_indicator.
  cc_indicator = population_constraints_completion
  
  # Save cc_indicator values in order to adjust cc_min.
  cc_indicator_values = []
  cc_indicator_values.append(cc_indicator)
  
  #print("START GUIDING")
  
  # While cc_indicator is below cc_min, apply variations to the population.
  while cc_indicator <= cc_min:
  
    #print("FORCE EXPLO - cc_ind=" + str(round(cc_indicator,2)))

    """
    # Remove 25% of the worst vectors from population.
    #print("==================================> REMOVE 25%")
    remove_quantity = int(0.25 * len(population))
    if remove_quantity < 1:
      remove_quantity = 1
    # Sort population by score DESC.
    population = sorted(deepcopy(population), key=score, reverse=True)
    # Remove
    for k in range(remove_quantity):
      population.pop()
    # Add in the population as many fresh generated vectors as the quantity removed.
    for k in range(remove_quantity):
      population.append(generate_vector())
    """
    for p in range(len(population)):
      # A 0-digit becomes a 1-digit while constraints are respected.
      done = False
      while done is False:
        rand = random.randint(0, len(population[p])-1)
        if population[p][rand] == 0:
          population[p][rand] = 1
          if constraints_max_values_check(population[p]) is False:
            done = True # End while loop.
            population[p][rand] = 0 # CANCEL LAST MODIFICATION

    # Adjust cc_min using the tolerance spread toward mean cc_indicator.
    mean_cc_indicator = mean(cc_indicator_values)
    cc_min = mean_cc_indicator - spread
    if cc_min < 0: 
      cc_min = 0

    # Update cc_indicator and add it to cc_indicator_values.
    # Calculate constraints completion.
    population_constraints_completion = 0
    vectors_constraints_completion = []
    for k in range(len(population)):
      digits_constraints_completion = []
      for d in range(len(population[k])):
        if population[k][d] == 1:
          digits_constraints_completion.append(digits_cc_scores[d])
      vectors_constraints_completion.append(sum(digits_constraints_completion))
    population_constraints_completion = mean(vectors_constraints_completion)
    cc_indicator = population_constraints_completion
    # Append to cc_indicator_values.
    cc_indicator_values.append(cc_indicator)


  # Normalise the population.
  for k in range(len(population)):
    population[k] = normalise(deepcopy(population[k]))
  
  return population



# Guide Exploration & Exploitation (E&E) With the links between digits.
# There is a link between two digits if they are both present in the same vector.
# Function to check if the balance between E&E is good regarding thresholds on a links gap indicator. If the balance is good, the function returns the population without modifications, else the function returns the population modified according to the E&E situation (force exploration or force exploitation). 
# Before being returned, the population is normalised.
# The population parameter must contain at least two vectors.
def guide_explo_with_links(population): 
  
  # Initialise the matrix which will contain the number of links between each digits couple.
  links = [[0 for i in range(len(population[0]))] for j in range(len(population[0]))]
  
  # Get 1-digit indexes from population vectors.
  one_digits_indexes = []
  for k in range(len(population)):
    vector_one_digits_indexes = []
    for d in range(len(population[k])):
      if population[k][d] == 1:
        vector_one_digits_indexes.append(d)
    one_digits_indexes.append(vector_one_digits_indexes)
  
  # In each vector, for each digits couple possible, increment the link value in links.
  for k in range(len(one_digits_indexes)):
    for index in range(len(one_digits_indexes[k])):
      for index2 in range(index+1, len(one_digits_indexes[k])):
        links[one_digits_indexes[k][index]][one_digits_indexes[k][index2]] += 1

  # Transform the links matrix into an array.
  links = array(links).flatten()
  
  # Remove zero values from links.
  links = links[links != 0]
  
  # Let's calculate our indicator named links_indicator.
  # Calculate the gap between the max and the mean number of links.
  # Calculate how many percents the gap represents toward the max number of links.
  mean_number_of_links = mean(links)
  max_number_of_links = max(links)
  gap = max_number_of_links - mean_number_of_links
  gap_percent = gap/max_number_of_links
  links_indicator = gap_percent
  
  # Save links_indicator values in order to adjust e_e_min and e_e_max.
  links_indicator_values = []
  links_indicator_values.append(links_indicator)
  
  # Starting limits min and max for links_indicator + tolerance spread around the mean links_indicator.
  e_e_min = 0.1
  e_e_max = 0.1
  spread = 0.1
  
  #print("START GUIDING")

  while links_indicator <= e_e_min or links_indicator >= e_e_max:

    while links_indicator <= e_e_min:
      
      #print("===> FORCE EXPLORATION - Indicator= " + str(round(links_indicator, 2)) + " ", end="\r")
      #print("===> FORCE EXPLORATION - Indicator= " + str(round(links_indicator, 2)))
      
      
      # Remove 25% of the worst vectors from population.
      #print("==================================> REMOVE 25%")
      remove_quantity = int(0.25 * len(population))
      if remove_quantity < 1:
        remove_quantity = 1
      # Sort population by score DESC.
      population = sorted(deepcopy(population), key=score, reverse=True)
      # Remove
      for k in range(remove_quantity):
        population.pop()
      # Add in the population as many fresh generated vectors as the quantity removed.
      for k in range(remove_quantity):
        population.append(generate_vector())
      
      """
      # In each vector of population, a 0-digit becomes a 1-digit.
      for k in range(len(population)):
        done = False
        while done is False:
          rand = random.randint(0, len(population[k])-1)
          if population[k][rand] == 0:
            population[k][rand] = 1
            done = True
      """
      # Adjust e_e_min and e_e_max using the tolerance spread toward mean links_indicator.
      mean_links_indicator = mean(links_indicator_values)
      e_e_min = mean_links_indicator - spread
      e_e_max = mean_links_indicator + spread
      
      # Update links_indicator and add it to links_indicator_values.
      links = [[0 for i in range(len(population[0]))] for j in range(len(population[0]))]
      one_digits_indexes = []
      for k in range(len(population)):
        vector_one_digits_indexes = []
        for d in range(len(population[k])):
          if population[k][d] == 1:
            vector_one_digits_indexes.append(d)
        one_digits_indexes.append(vector_one_digits_indexes)
      for k in range(len(one_digits_indexes)):
        for index in range(len(one_digits_indexes[k])):
          for index2 in range(index+1, len(one_digits_indexes[k])):
            links[one_digits_indexes[k][index]][one_digits_indexes[k][index2]] += 1
      links = array(links).flatten()
      links = links[links != 0]
      mean_number_of_links = mean(links)
      max_number_of_links = max(links)
      gap = max_number_of_links - mean_number_of_links
      gap_percent = gap/max_number_of_links
      links_indicator = gap_percent
      links_indicator_values.append(links_indicator)
  
  
    while links_indicator >= e_e_max:

      #print("===> FORCE EXPLOITATION - Indicator= " + str(round(links_indicator, 2)) + " ", end="\r")
      #print("===> FORCE EXPLOITATION - Indicator= " + str(round(links_indicator, 2)))
      
      # In each vector of population, a 0-digit becomes a 1-digit while constraints are respected.
      for k in range(len(population)):
        done = False
        while done is False:
          rand = random.randint(0, len(population[k])-1)
          if population[k][rand] == 0:
            population[k][rand] = 1
            if constraints_max_values_check(population[k]) is False:
              done = True # End while loop.
              population[k][rand] = 0 # CANCEL LAST MODIFICATION
      
      # Adjust e_e_min and e_e_max using the tolerance spread toward mean links_indicator.
      mean_links_indicator = mean(links_indicator_values)
      e_e_min = mean_links_indicator - spread
      e_e_max = mean_links_indicator + spread
      
      # Update links_indicator and add it to links_indicator_values.
      links = [[0 for i in range(len(population[0]))] for j in range(len(population[0]))]
      one_digits_indexes = []
      for k in range(len(population)):
        vector_one_digits_indexes = []
        for d in range(len(population[k])):
          if population[k][d] == 1:
            vector_one_digits_indexes.append(d)
        one_digits_indexes.append(vector_one_digits_indexes)
      for k in range(len(one_digits_indexes)):
        for index in range(len(one_digits_indexes[k])):
          for index2 in range(index+1, len(one_digits_indexes[k])):
            links[one_digits_indexes[k][index]][one_digits_indexes[k][index2]] += 1
      links = array(links).flatten()
      links = links[links != 0]
      mean_number_of_links = mean(links)
      max_number_of_links = max(links)
      gap = max_number_of_links - mean_number_of_links
      gap_percent = gap/max_number_of_links
      links_indicator = gap_percent
      links_indicator_values.append(links_indicator)


  # Normalise the population.
  for k in range(len(population)):
    population[k] = normalise(deepcopy(population[k]))
  
  return population




# Guide Exploration & Exploitation (E&E) With the number of links between digits inside each vector.
# There is a link between two digits if they are both present in the same vector.
# Function to check if the balance between E&E is good regarding thresholds on a links gap indicator between vectors. If the balance is good, the function returns the population without modifications, else the function returns the population modified according to the E&E situation (force exploration or force exploitation). 
# Before being returned, the population is normalised.
# The population parameter must contain at least two vectors.
def guide_explo_with_vector_links(population):
  
  # Initialise the array which will contain the number of links in each vector.
  links = []
  
  # Get the number of 1-digits in each vector, calculate the number of link possibilities and append it to links.
  for k in range(len(population)):
    v_one_digit_number = 0
    for d in range(len(population[k])):
      if population[k][d] == 1:
        v_one_digit_number += 1
    link_possibilities = (v_one_digit_number*(v_one_digit_number-1))/2
    links.append(link_possibilities)

  # Calculate our indicator named vector_links_indicator.
  # Calculate the gap between the max and the mean of links values.
  # Calculate how many percents the gap represents toward the max of links values.
  mean_links = mean(links)
  max_links = max(links)
  gap = max_links - mean_links
  gap_percent = gap/max_links
  vector_links_indicator = gap_percent
  
  # Save vector_links_indicator values in order to adjust e_e_min and e_e_max.
  vector_links_indicator_values = []
  vector_links_indicator_values.append(vector_links_indicator)
  
  # Starting limits min and max for vector_links_indicator + tolerance spread around the mean vector_links_indicator.
  e_e_min = 0.1
  e_e_max = 0.1
  spread = 0.1
  
  #print("START GUIDING")

  while vector_links_indicator <= e_e_min or vector_links_indicator >= e_e_max:

    while vector_links_indicator <= e_e_min:

      #print("===> FORCE EXPLORATION - Indicator= " + str(round(vector_links_indicator, 2)) + " ", end="\r")
      #print("===> FORCE EXPLORATION - Indicator= " + str(round(vector_links_indicator, 2)))
      
      """
      # Remove 25% of the worst vectors from population.
      #print("==================================> REMOVE 25%")
      remove_quantity = int(0.25 * len(population))
      if remove_quantity < 1:
        remove_quantity = 1
      # Sort population by score DESC.
      population = sorted(deepcopy(population), key=score, reverse=True)
      # Remove
      for k in range(remove_quantity):
        population.pop()
      # Add in the population as many fresh generated vectors as the quantity removed.
      for k in range(remove_quantity):
        population.append(generate_vector())
      """
      
      # In each vector of population, a 0-digit becomes a 1-digit.
      for k in range(len(population)):
        done = False
        while done is False:
          rand = random.randint(0, len(population[k])-1)
          if population[k][rand] == 0:
            population[k][rand] = 1
            done = True
      
      
      # Adjust e_e_min and e_e_max using the tolerance spread toward mean vector_links_indicator.
      mean_vector_links_indicator = mean(vector_links_indicator_values)
      e_e_min = mean_vector_links_indicator - spread
      e_e_max = mean_vector_links_indicator + spread
      
      # Update vector_links_indicator and add it to vector_links_indicator_values.
      for k in range(len(population)):
        v_one_digit_number = 0
        for d in range(len(population[k])):
          if population[k][d] == 1:
            v_one_digit_number += 1
        link_possibilities = (v_one_digit_number*(v_one_digit_number-1))/2
        links.append(link_possibilities)
      mean_links = mean(links)
      max_links = max(links)
      gap = max_links - mean_links
      gap_percent = gap/max_links
      vector_links_indicator = gap_percent
      vector_links_indicator_values.append(vector_links_indicator)


    while vector_links_indicator >= e_e_max:

      #print("===> FORCE EXPLOITATION - Indicator= " + str(round(vector_links_indicator, 2)) + " ", end="\r")
      #print("===> FORCE EXPLOITATION - Indicator= " + str(round(vector_links_indicator, 2)))
      
      # In each vector of population, a 0-digit becomes a 1-digit while constraints are respected.
      for k in range(len(population)):
        done = False
        while done is False:
          rand = random.randint(0, len(population[k])-1)
          if population[k][rand] == 0:
            population[k][rand] = 1
            if constraints_max_values_check(population[k]) is False:
              done = True # End while loop.
              population[k][rand] = 0 # CANCEL LAST MODIFICATION

      # Adjust e_e_min and e_e_max using the tolerance spread toward mean vector_links_indicator.
      mean_vector_links_indicator = mean(vector_links_indicator_values)
      e_e_min = mean_vector_links_indicator - spread
      e_e_max = mean_vector_links_indicator + spread
      
      # Update vector_links_indicator and add it to vector_links_indicator_values.
      for k in range(len(population)):
        v_one_digit_number = 0
        for d in range(len(population[k])):
          if population[k][d] == 1:
            v_one_digit_number += 1
        link_possibilities = (v_one_digit_number*(v_one_digit_number-1))/2
        links.append(link_possibilities)
      mean_links = mean(links)
      max_links = max(links)
      gap = max_links - mean_links
      gap_percent = gap/max_links
      vector_links_indicator = gap_percent
      vector_links_indicator_values.append(vector_links_indicator)
      

  # Normalise the population.
  for k in range(len(population)):
    population[k] = normalise(deepcopy(population[k]))
  
  return population




# Guide Exploration & Exploitation (E&E) With various indicators tagged as "E&E Power".
# - Counters of iterations that allowed to discover new digits          (exploration) and that didn't allow to discover new digits (exploitation).
# - Counters of vectors that allowed to discover new digits (exploration) and that didn't allow to discover new digits (exploitation).
# - Counters of newly discovered digits and of already discovered digits.
# The function checks if the balance between E&E is good regarding thresholds on these indicators. If the balance is good, the function returns the population without modifications, else the function returns the population modified according to the E&E situation (force exploration or force exploitation). 
# Before being returned, the population is normalised.
# The population parameter must contain at least two vectors.
def guide_explo_with_power(population):

  # Condition to avoid division by zero error.
  if counter_exploit_iter > 0:
    # Calculate the indicator.
    # The indicator will be the mean of three percentages calculated with the six E&E power global variables.
    # NOTA : percentage_3 is multiplied by 1000 to make its value in the same height order as the two other percentages.
    percentage_1 = counter_explore_iter / counter_exploit_iter
    percentage_2 = counter_explore_vector / counter_exploit_vector
    percentage_3 = (counter_explore_digit / counter_exploit_digit)*1000
    ee_power_indicator = mean([percentage_1, percentage_2, percentage_3])
    
    # Save indicator values in order to adjust e_e_min and e_e_max.
    ee_power_indicator_values = []
    ee_power_indicator_values.append(ee_power_indicator)
    
    # Starting limits min and max for the indicator + tolerance spread around the mean indicator.
    e_e_min = 0.05
    e_e_max = 0.05
    spread = 0.01
    
    # Counters of values to add to E&E power variables when we update the percentages to update the indicator (see inside while loops below).
    # Everytime we force an exploration or exploitation, we will increment some of those counters.
    add_to_counter_explore_iter = 0
    add_to_counter_exploit_iter = 0
    add_to_counter_explore_vector = 0
    add_to_counter_exploit_vector = 0
    add_to_counter_explore_digit = 0
    add_to_counter_exploit_digit = 0
    
    #print("START GUIDING")

    while ee_power_indicator <= e_e_min or ee_power_indicator >= e_e_max:

      while ee_power_indicator <= e_e_min:

        #print("===> FORCE EXPLORATION - Indicator= " + str(round(ee_power_indicator, 2)) + " ", end="\r")
        #print("===> FORCE EXPLORATION - Indicator= " + str(round(ee_power_indicator, 2)))
        
        add_to_counter_explore_iter += 1
        
        # Do variations to force exploration.
        # For each vector in the population, we randomly select a digit index from ndd (non-discovered digits) and we put it to 1 in the vector. If ndd is empty, we randomly select a digit index from dd (already discovered digits).
        for k in range(len(population)):
          rand_index = 0
          if len(ndd) > 0:
            rand_index = random.choice(ndd)
          else:
            rand_index = random.choice(dd)
          population[k][rand_index] = 1
          add_to_counter_explore_vector += 1
          add_to_counter_explore_digit += 1
          
        # Adjust e_e_min and e_e_max using the tolerance spread toward mean ee_power_indicator.
        mean_ee_power_indicator = mean(ee_power_indicator_values)
        e_e_min = mean_ee_power_indicator - spread
        e_e_max = mean_ee_power_indicator + spread
        
        # Update ee_power_indicator and add it to ee_power_indicator_values.
        percentage_1 = (counter_explore_iter + add_to_counter_explore_iter) / (counter_exploit_iter + add_to_counter_exploit_iter)
        percentage_2 = (counter_explore_vector + add_to_counter_explore_vector) / (counter_exploit_vector + add_to_counter_exploit_vector)
        percentage_3 = ((counter_explore_digit + add_to_counter_explore_digit) / (counter_exploit_digit + add_to_counter_exploit_digit))*1000
        ee_power_indicator = mean([percentage_1, percentage_2, percentage_3])
        ee_power_indicator_values.append(ee_power_indicator)


      while ee_power_indicator >= e_e_max:

        #print("===> FORCE EXPLOITATION - Indicator= " + str(round(ee_power_indicator, 2)) + " ", end="\r")
        #print("===> FORCE EXPLOITATION - Indicator= " + str(round(ee_power_indicator, 2)))
        
        add_to_counter_exploit_iter += 1
        
        # Do variations to force exploitation.
        # For each vector in the population, we randomly select a digit index from dd (already discovered digits) and we put it to 1 in the vector.
        for k in range(len(population)):
          rand_index = random.choice(dd)
          population[k][rand_index] = 1
          add_to_counter_exploit_vector += 1
          add_to_counter_exploit_digit += 1
          
        # Adjust e_e_min and e_e_max using the tolerance spread toward mean ee_power_indicator.
        mean_ee_power_indicator = mean(ee_power_indicator_values)
        e_e_min = mean_ee_power_indicator - spread
        e_e_max = mean_ee_power_indicator + spread
        
        # Update ee_power_indicator and add it to ee_power_indicator_values.
        percentage_1 = (counter_explore_iter + add_to_counter_explore_iter) / (counter_exploit_iter + add_to_counter_exploit_iter)
        percentage_2 = (counter_explore_vector + add_to_counter_explore_vector) / (counter_exploit_vector + add_to_counter_exploit_vector)
        percentage_3 = ((counter_explore_digit + add_to_counter_explore_digit) / (counter_exploit_digit + add_to_counter_exploit_digit))*1000
        ee_power_indicator = mean([percentage_1, percentage_2, percentage_3])
        ee_power_indicator_values.append(ee_power_indicator)
        

    # Normalise the population.
    for k in range(len(population)):
      population[k] = normalise(deepcopy(population[k]))
  
  return population




# Guide Exploration & Exploitation (E&E) With various indicators tagged as "E&E Temporal Balance".
# The function checks if the balance between E&E is good regarding thresholds on these indicators. If the balance is good, the function returns the population without modifications, else the function returns the population modified according to the E&E situation (force exploration or force exploitation). 
# Before being returned, the population is normalised.
# The population parameter must contain at least two vectors.
def guide_explo_with_temporal_balance(population):
  
  # We must have at least 3 timers in explore_iter_timers to get 2 gaps.
  if len(explore_iter_timers) > 2:
    
    # Calculate the indicator.
    # The indicator will be the mean gap between exploration iterations.
    # To do this :
    # Calculate gaps between successive timers from explore_iter_timers.
    # We do not consider the starting algorithm execution timer (saet timer) and the last algorithm execution timer (laet timer) because we consider that the most important is to analyse the core of the E&E behaviour from where it started to where it ended. We justify that by the fact that the gap between the saet and the first exploration timer could take a completely absurd value, same thing for the gap between the last exploration timer and the laet.
    e_i_timers_gaps = []
    for k in range(len(explore_iter_timers)-1):
      gap = explore_iter_timers[k+1] - explore_iter_timers[k]
      e_i_timers_gaps.append(gap)
    # Min gap
    min_eit_gap = min(e_i_timers_gaps)
    # Max gap
    max_eit_gap = max(e_i_timers_gaps)
    # Mean of min_gap and max_gap.
    mean_of_min_max_eit_gaps = (min_eit_gap + max_eit_gap)/2
    # Mean gap
    mean_eit_gap = mean(e_i_timers_gaps)
    # Indicator
    ee_tb_indicator = mean_eit_gap
    
    #print("START GUIDING")

    if ee_tb_indicator >= mean_of_min_max_eit_gaps:

      #print("===> FORCE EXPLORATION - Indicator= " + str(round(ee_power_indicator, 2)) + " ", end="\r")
      #print("===> FORCE EXPLORATION - Indicator= " + str(round(ee_tb_indicator, 2)))
      
      """
      # Remove 25% of the worst vectors from population.
      #print("==================================> REMOVE 25%")
      remove_quantity = int(0.25 * len(population))
      if remove_quantity < 1:
        remove_quantity = 1
      # Sort population by score DESC.
      population = sorted(deepcopy(population), key=score, reverse=True)
      # Remove
      for k in range(remove_quantity):
        population.pop()
      # Add in the population as many fresh generated vectors as the quantity removed.
      for k in range(remove_quantity):
        population.append(generate_vector())
      """
      # In each vector of population, a 0-digit becomes a 1-digit.
      for k in range(len(population)):
        done = False
        while done is False:
          rand = random.randint(0, len(population[k])-1)
          if population[k][rand] == 0:
            population[k][rand] = 1
            done = True     

    else:

      #print("===> FORCE EXPLOITATION - Indicator= " + str(round(ee_power_indicator, 2)) + " ", end="\r")
      #print("===> FORCE EXPLOITATION - Indicator= " + str(round(ee_tb_indicator, 2)))
      
      # In each vector of population, a 0-digit becomes a 1-digit while constraints are respected.
      for k in range(len(population)):
        done = False
        while done is False:
          rand = random.randint(0, len(population[k])-1)
          if population[k][rand] == 0:
            population[k][rand] = 1
            if constraints_max_values_check(population[k]) is False:
              done = True # End while loop.
              population[k][rand] = 0 # CANCEL LAST MODIFICATION


    # Normalise the population.
    for k in range(len(population)):
      population[k] = normalise(deepcopy(population[k]))
  
  return population








# Random
def random_():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  
  pop = []
  population_size = 4
  
  strongest_vector = []
  max_score = 0
  iteration_score_evolution = []
  max_score_evolution = []
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0
  
  
  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations
    
    print("Iteration N° " + str(i+1) + " ", end="\r")
    
    # Reset the population.
    pop = []
    for k in range(population_size):
      vector = generate_vector()
      pop.append({
                    'vector':vector,
                    'score':score(vector)
                })

    # --------------------
    # GUIDE E&E.
    if ee_guided is True:
      # Get the vectors.
      vectors = []
      for k in range(len(pop)):
        vectors.append(pop[k]['vector'])
      # Guide.
      vectors = guide_explo(vectors)
      # Update population vectors and scores.
      for k in range(len(vectors)):
        pop[k]['vector'] = vectors[k]
        pop[k]['score'] = score(vectors[k])
    # --------------------

    # Sort the population by score DESC.  
    pop = sorted(deepcopy(pop), key=lambda dct: dct['score'], reverse=True)
    
    # Add to top_vectors and all_vectors.
    for k in range(len(pop)):
      vector_score = pop[k]['score']
      add_to_top_vectors(pop[k]['vector'], vector_score)   
      all_vectors.append(pop[k]['vector'])

    # EVALUATIONS - ITERATION END 
    iteration_score = pop[0]['score']
    iteration_score_evolution.append(iteration_score)
    
    if iteration_score > max_score:
      max_score = iteration_score
      strongest_vector = deepcopy(pop[0]['vector'])
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:
      # Convergence Management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score
        
    max_score_evolution.append(max_score)

  if convergence_score == 0: # if convergence not reached, the convergence is considered to be the max_score.
    convergence_iteration = max_score_iteration
    convergence_score = max_score
    
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score, 
            res
          ]


# Reset "dd", "ndd" and E&E Power counters.
# Function to reset the discovered digits array and the non-discovered digits array and the various E&E counters.
def reset_dd_ndd_counters():
  global dd
  global ndd
  global counter_explore_iter
  global counter_exploit_iter
  global counter_explore_vector
  global counter_exploit_vector
  global counter_explore_digit
  global counter_exploit_digit
  # Reset "dd".
  dd = []
  # Reset "ndd".
  ndd = []
  for k in range(len(df)):
    ndd.append(k)
  # Reset counters
  counter_explore_iter = 0
  counter_exploit_iter = 0
  counter_explore_vector = 0
  counter_exploit_vector = 0
  counter_explore_digit = 0
  counter_exploit_digit = 0
    
# Transfer 1-digits in vector, still present in "ndd", from "ndd" to "dd".
def transfer_from_ndd_to_dd(vector):
  global dd
  global ndd
  for d in range(len(vector)):
    if vector[d] == 1 and d in ndd:
      ndd.remove(d)
      dd.append(d)
  
# Update E&E Power counters.
def feed_ee_counters(vectors):
  global counter_explore_iter
  global counter_exploit_iter
  global counter_explore_vector
  global counter_exploit_vector
  global counter_explore_digit
  global counter_exploit_digit
  global explore_iter_timers
  
  check_iter_centered_on_exploration = 0
  for k in range(len(vectors)):
    check_in_ndd = 0
    for d in range(len(vectors[k]["vector"])):
      if vectors[k]["vector"][d] == 1 and d in ndd:
        check_in_ndd += 1
        check_iter_centered_on_exploration += 1
        counter_explore_digit += 1
      else:
        counter_exploit_digit += 1
    if check_in_ndd > 0:
      counter_explore_vector += 1
    else:
      counter_exploit_vector += 1
  
    # Transfer newly discovered digits indexes from "ndd" to "dd".
    transfer_from_ndd_to_dd(deepcopy(vectors[k]['vector']))
    
  # Whether the iteration has explored or not, we increment the corresponding iteration counter.
  if check_iter_centered_on_exploration > 0:
    counter_explore_iter += 1
    explore_iter_timers.append(time.perf_counter())
  else:
    counter_exploit_iter += 1


# Genetic
# Classic Genetic Algorithm used as comparison reference in experimentations.
def genetic():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []

  pop = []
  population_size = 4
  
  # Strongest vector
  v_strong = []
  # Score of the strongest vector
  max_score = 0
  
  iteration_score_evolution = []
  each_generation_mean_score = []
  max_score_evolution = []
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0
    
  
  # INITIALISATION
  # Each vector contains as many digits as pairs action-resource in df.
  for k in range(population_size):
    vector = generate_vector()
    vector_score = score(vector)
    pop.append({
                  'vector':vector,
                  'score':vector_score
              })
    # Add in top_vectors and all_vectors.
    add_to_top_vectors(deepcopy(vector), vector_score)   
    all_vectors.append(deepcopy(vector))
    # Transfer newly discovered digits indexes from "ndd" to "dd".
    transfer_from_ndd_to_dd(deepcopy(vector))


  generation_value = algorithm_iterations
  for i in range(generation_value): # x iterations
    
    print("Iteration N° " + str(i+1) + " ", end="\r")

    # PARENTS SELECTION + REPRODUCTION
    # Number of couples to create = population_size/2
    # Take 2 parents randomly from the population and make them reproduce.
    # Two parents make two children.
    # To perform a reproduction, we take the left part of the considered parent (until a random hybridation index). We stick the right part of the other parent (from the hybridation index) to the considered parent.
    parents = []
    childs = []
    number_of_couples = int(population_size/2)
    for k in range(number_of_couples):
      # 1st parent
      index = random.randint(0, len(pop)-1)
      parent_1 = pop[index]
      pop.pop(index)
      parents.append(parent_1)
      # 2nd parent
      index = random.randint(0, len(pop)-1)
      parent_2 = pop[index]
      pop.pop(index)
      parents.append(parent_2)
      # 1st child
      child_1 = deepcopy(parent_1)
      hybridation_index = random.randint(0, len(df)-1)
      for j in range(hybridation_index, len(df)):
        child_1['vector'][j] = parent_2['vector'][j]
      childs.append(child_1)
      # 2nd child
      child_2 = deepcopy(parent_2)
      hybridation_index = random.randint(0, len(df)-1)
      for j in range(hybridation_index, len(df)):
        child_2['vector'][j] = parent_1['vector'][j]
      childs.append(child_2)

    # MUTATIONS
    for k in range(len(childs)):
      # While childs[k]['vector'] respects max_value constraints...
      while constraints_max_values_check(childs[k]['vector']) is True:
        # ...Randomly transform a 0-digit into a 1-digit in childs[k]['vector'].
        rand = random.randint(0, len(df)-1)
        childs[k]['vector'][rand] = 1
      # NORMALISATION
      childs[k]['vector'] = normalise(deepcopy(childs[k]['vector']))


    # --------------------
    # GUIDE E&E.
    if ee_guided is True:
      # Get the vectors.
      vectors = []
      for c in range(len(childs)):
        vectors.append(deepcopy(childs[c]['vector']))
      
      # Guide.
      if guide_function == "separated_EE_cases":
        vectors = guide_explo_with_separated_EE_cases(deepcopy(vectors))
      elif guide_function == "merged_EE_cases":
        vectors = guide_explo_with_merged_EE_cases(deepcopy(vectors))
      elif guide_function == "potential":
        vectors = guide_explo_with_potential(deepcopy(vectors))
      elif guide_function == "cc":
        vectors = guide_explo_with_cc(deepcopy(vectors))
      elif guide_function == "links":
        vectors = guide_explo_with_links(deepcopy(vectors))
      elif guide_function == "vector_links":
        vectors = guide_explo_with_vector_links(deepcopy(vectors))
      elif guide_function == "power":
        vectors = guide_explo_with_power(deepcopy(vectors))
      elif guide_function == "temporal_balance":
        vectors = guide_explo_with_temporal_balance(deepcopy(vectors))
        
      # Update childs vectors and scores.
      for v in range(len(vectors)):
        childs[v]['vector'] = deepcopy(vectors[v])
    # --------------------   
    
    
    # LAST TREATMENTS
    # Feed E&E counters
    feed_ee_counters(deepcopy(childs))
    # Other last treatments
    for k in range(len(childs)):
      # Update score
      childs[k]['score'] = score(deepcopy(childs[k]['vector']))
      # Add to top_vectors and all_vectors.
      add_to_top_vectors(deepcopy(childs[k]['vector']), deepcopy(childs[k]['score']))
      all_vectors.append(deepcopy(childs[k]['vector']))

    
    # EVALUATIONS - ITERATION END
    
    # Population for next iteration will be constituted of the best vectors in reunited parents and childs.
    pop = []
    pop = deepcopy(parents) + deepcopy(childs)
    # Sort population by score DESC.
    pop = sorted(deepcopy(pop), key=lambda dct: dct['score'], reverse=True)
    # Keep the best vectors by removing the last ones until population size is respected.
    while len(pop) > population_size:
      pop.pop() # Remove last element from population (weakest one)    
    
    # Strongest vector (v_strong) and associated score (max_score)
    over_max_score = False
    if pop[0]['score'] > max_score:
      over_max_score = True
      max_score = pop[0]['score']
      v_strong = deepcopy(pop[0]['vector'])

    iteration_score_evolution.append(pop[0]['score'])
    
    mean_score_pop = 0
    for k in range(len(pop)):
      mean_score_pop += pop[k]['score']
    mean_score_pop /= len(pop)
    
    each_generation_mean_score.append(mean_score_pop)
    
    if over_max_score is True:
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:
      # Convergence Management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score
    
    max_score_evolution.append(max_score)

  if convergence_score == 0: # if convergence not yet reached, convergence is considered to be max_score.
    convergence_iteration = max_score_iteration
    convergence_score = max_score
    
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score, 
            res
          ]



# Update E&E Power counters. LOGGED VERSION
def feed_ee_counters_logged(vectors, cause, iter):
  global counter_explore_iter
  global counter_exploit_iter
  global counter_explore_vector
  global counter_exploit_vector
  global counter_explore_digit
  global counter_exploit_digit
  global explore_iter_timers
  
  check_iter_centered_on_exploration = 0
  for k in range(len(vectors)):
    check_in_ndd = 0
    for d in range(len(vectors[k]["vector"])):
      if vectors[k]["vector"][d] == 1 and d in ndd:
        check_in_ndd += 1
        check_iter_centered_on_exploration += 1
        counter_explore_digit += 1
      else:
        counter_exploit_digit += 1
    if check_in_ndd > 0:
      counter_explore_vector += 1
    else:
      counter_exploit_vector += 1
  
    """
    # Feed ee_logs.
    ee_type = 'Exploitation'
    if check_in_ndd > 0:
      ee_type = 'Exploration'
    ee_logs[iter]['vectors'].append({
                    'vector':vectors[k]["vector"],
                    'ee_type':ee_type,
                    'number_of_new_digits':check_in_ndd,
                    'cause':cause,
                    'timer':round(time.perf_counter(), 2)
                  })
    """
    
  for k in range(len(vectors)):  
    # Transfer newly discovered digits indexes from "ndd" to "dd".
    transfer_from_ndd_to_dd(deepcopy(vectors[k]['vector']))
    
  # Whether the iteration has explored or not, we increment the corresponding iteration counter.
  if check_iter_centered_on_exploration > 0:
    counter_explore_iter += 1
    explore_iter_timers.append(time.perf_counter())
  else:
    counter_exploit_iter += 1

  # Feed remaining attributes from the iteration log.
  finalise_iteration_log(iter, vectors)


# Function to feed remaining attributes of a specific iteration log.
def finalise_iteration_log(iteration_number, vectors):
  global counter_exploit_iter
  global counter_explore_iter
  
  # Compute vectors_quantity. Quantity of generated vectors.
  vectors_quantity = len(all_vectors)
  
  
  # Compute tq_o. Number of times that the most used digit (taking value 1) has been used.
  summed = sum(all_vectors, axis=0)
  tq_o = max(summed)
  # Compute mq_o. Mean number of times that used digits has been used.
  used_ones_quantity = summed[summed != 0] # Remove 0 from summed.
  mq_o = mean(used_ones_quantity)
  
  # Get mean_values of resources present in summed.
  present_mean_values = []
  for k in range(len(summed)):
    if summed[k] > 0:
      present_mean_values.append(df_mean_values[k])  
  # Compute tq_v. Mean value of the used digit having the best mean value.
  tq_v = max(present_mean_values)
  # Compute mq_v. Mean of mean values of used digits.
  mq_v = mean(present_mean_values)


  # Compute coverage in all_vectors.
  all_cov = len(used_ones_quantity)/len(summed)

  # Compute power. This is the Exploration Power which represents the portion of exploration iterations toward current elapsed iterations.
  power = counter_explore_iter / (iteration_number+1)

  # Get the score of the vector having the best one from top_vectors.
  scores = []
  for k in range(len(top_vectors)):
    scores.append(top_vectors[k]["score"])
  top_vectors_score = max(scores)
  
  # Get the constraints completion of the vector having the best one from top_vectors.
  all_constraints_completions = []
  for k in range(len(top_vectors)):  
    # Get the constraints completions of the vector.
    vector_constraints_values = []
    for c in range(len(constraints)):
      vector_constraints_values.append(0)
    for i in range(len(top_vectors[k]['vector'])):
      if top_vectors[k]['vector'][i] == 1:
        for c in range(len(constraints)):
          vector_constraints_values[c] += df[i][constraints[c][0]]
    # Calculate the mean constraints completion of this vector.
    const_comp_values = []
    for c in range(len(vector_constraints_values)):
      const_comp_values.append(vector_constraints_values[c]/constraints[c][1])
    vector_mean_const_comp = mean(const_comp_values)
    all_constraints_completions.append(vector_mean_const_comp)
  top_vectors_constraints_completion = max(all_constraints_completions)
  
  # Get the mean similarity of iteration vectors with top_vectors.
  similarities = []
  for k in range(len(vectors)):
    for tv in range(len(top_vectors)):
      common_one_digits = 0
      for d in range(len(vectors[k]['vector'])):
        if vectors[k]['vector'][d] == 1 and top_vectors[tv]['vector'][d] == 1:
          common_one_digits += 1
      similarity = common_one_digits/sum(vectors[k]['vector'])
      similarities.append(similarity)
  mean_similarity = mean(similarities)



  
  
  # Create the pool of already discovered resources
  pool = []
  for k in range(len(summed)):
    if summed[k] > 0:
      to_append = deepcopy(df[k])
      to_append[0] = float(to_append[0])
      pool.append(to_append)  
  
  # Compute HVV = HVVd/HVVmax.
  dimensions_values_portions = []
  # Take the max value of each column of the pool (except the first column which is the resource id). Multiply these max values to get HVVd.
  pool_maxes = np_max(array(pool), axis=0)
  hvvd = 1
  for k in range(1, len(pool_maxes)):
    qty = pool_maxes[k]
    hvvd *= qty
    # Add to dimensions_values_portions, the fraction "qty/corresponding dimensions_values_max".
    dimensions_values_portions.append(round(qty/dimensions_values_max[k-1], 2))
  # Do the fraction to get HVV = HVVd/HVVmax.
  hvv = hvvd/hvv_max
  
  # Compute HVO = HVOd/HVOmax.
  dimensions_occurrences_portions = []
  pool_matrix = np_matrix(pool)
  pool_transposed = pool_matrix.T
  hvod = 1
  for k in range(1, len(pool_transposed)):
    # Get the unique values of the dimension considered by converting the list to a set. Calculate its len and multiply HVOd by it.
    qty = len(set(pool_transposed[k].tolist()[0]))
    hvod *= qty
    # Add to dimensions_occurrences_portions, the fraction "qty/corresponding dimensions_occurrences_max".
    dimensions_occurrences_portions.append(round(qty/dimensions_occurrences_max[k-1], 2))
  # Do the fraction to get HVO = HVOd/HVOmax.
  hvo = hvod/hvo_max
  
  # Compute HVCO and HVCV.
  hvco = (((mq_o + tq_o)/2) * hvo_max) / hvod
  hvcv = (((mq_v + tq_v)/2) * hvv_max) / hvvd

  
  
  # Feed iteration log attributes.
  ee_logs[iteration_number]['vectors_quantity'] = vectors_quantity
  #ee_logs[iteration_number]['top_quantity'] = top_quantity
  ee_logs[iteration_number]['power'] = round(power, 2)
  ee_logs[iteration_number]['top_vectors_score'] = round(top_vectors_score, 2)
  ee_logs[iteration_number]['all_vectors_coverage'] = round(all_cov, 2)
  ee_logs[iteration_number]['top_vectors_constraints_completion'] = round(top_vectors_constraints_completion, 2)
  ee_logs[iteration_number]['similarity_with_top_vectors'] = round(mean_similarity, 2)  
  ee_logs[iteration_number]['hvv'] = round(hvv, 2)
  ee_logs[iteration_number]['hvo'] = round(hvo, 2)
  ee_logs[iteration_number]['d%v'] = dimensions_values_portions
  ee_logs[iteration_number]['d%o'] = dimensions_occurrences_portions
  ee_logs[iteration_number]['hvcv'] = round(hvcv, 2)
  ee_logs[iteration_number]['hvco'] = round(hvco, 2)
  



# Genetic E&E Logged
# Classic Genetic Algorithm with information saved as logs regarding its E&E behaviour.
def genetic_ee_logged():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []

  pop = []
  population_size = 4
  
  # Strongest vector
  v_strong = []
  # Score of the strongest vector
  max_score = 0
  
  iteration_score_evolution = []
  each_generation_mean_score = []
  max_score_evolution = []
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0
    
  
  # INITIALISATION
  # Create the log structure for the iteration (here it is iteration number 0 because we are initializing the population).
  ee_logs.append({
                'iteration':0,
                'vectors':[],
                'vectors_quantity':0,
                'top_quantity':0,
                'top_vectors_score':0,
                'all_vectors_coverage':0,
                'top_vectors_constraints_completion':0,
                'similarity_with_top_vectors':0,
                'power':0,
                'hvv':0,
                'hvo':0,
                'd%v':[],
                'd%o':[],
                'hvcv':0,
                'hvco':0
              })
  # Create a number of vector equal to the population size and add them into the iteration log structure in 'vectors'.
  # Each vector contains as many digits as pairs action-resource in df.
  for k in range(population_size):
    vector = generate_vector()
    vector_score = score(vector)
    pop.append({
                  'vector':vector,
                  'score':vector_score
              })
    # Add in top_vectors and all_vectors.
    add_to_top_vectors(deepcopy(vector), vector_score)   
    all_vectors.append(deepcopy(vector))
    # Add to vectors_classes.
    add_to_vectors_classes(deepcopy(vector), 0)
    
  # Feed ee_counters and the other attributes of the iteration log.
  feed_ee_counters_logged(deepcopy(pop), 'Pure Random Function', 0)


  generation_value = algorithm_iterations
  for i in range(generation_value): # x iterations
    
    print("Iteration N° " + str(i+1) + " ", end="\r")

    # PARENTS SELECTION + REPRODUCTION
    # Number of couples to create = population_size/2
    # Take 2 parents randomly from the population and make them reproduce.
    # Two parents make two children.
    # To perform a reproduction, we take the left part of the considered parent (until a random hybridation index). We stick the right part of the other parent (from the hybridation index) to the considered parent.
    parents = []
    childs = []
    number_of_couples = int(population_size/2)
    for k in range(number_of_couples):
      # 1st parent
      index = random.randint(0, len(pop)-1)
      parent_1 = pop[index]
      pop.pop(index)
      parents.append(parent_1)
      # 2nd parent
      index = random.randint(0, len(pop)-1)
      parent_2 = pop[index]
      pop.pop(index)
      parents.append(parent_2)
      # 1st child
      child_1 = deepcopy(parent_1)
      hybridation_index = random.randint(0, len(df)-1)
      for j in range(hybridation_index, len(df)):
        child_1['vector'][j] = parent_2['vector'][j]
      childs.append(child_1)
      # 2nd child
      child_2 = deepcopy(parent_2)
      hybridation_index = random.randint(0, len(df)-1)
      for j in range(hybridation_index, len(df)):
        child_2['vector'][j] = parent_1['vector'][j]
      childs.append(child_2)

    # MUTATIONS
    for k in range(len(childs)):
      # While childs[k]['vector'] respects max_value constraints...
      while constraints_max_values_check(childs[k]['vector']) is True:
        # ...Randomly transform a 0-digit into a 1-digit in childs[k]['vector'].
        rand = random.randint(0, len(df)-1)
        childs[k]['vector'][rand] = 1
      # NORMALISATION
      childs[k]['vector'] = normalise(deepcopy(childs[k]['vector']))

    
    # LAST TREATMENTS
    # Create the log structure for the iteration.
    ee_logs.append({
                  'iteration':i+1,
                  'vectors':[],
                  'vectors_quantity':0,
                  'top_quantity':0,
                  'top_vectors_score':0,
                  'all_vectors_coverage':0,
                  'top_vectors_constraints_completion':0,
                  'similarity_with_top_vectors':0,
                  'power':0,
                  'hvv':0,
                  'hvo':0,
                  'd%v':[],
                  'd%o':[],
                  'hvcv':0,
                  'hvco':0
                })
    # Scores + adds
    for k in range(len(childs)):
      # Update score
      childs[k]['score'] = score(deepcopy(childs[k]['vector']))
      # Add to top_vectors and all_vectors.
      add_to_top_vectors(deepcopy(childs[k]['vector']), deepcopy(childs[k]['score']))
      all_vectors.append(deepcopy(childs[k]['vector']))
      # Add to vectors_classes.
      add_to_vectors_classes(deepcopy(childs[k]['vector']), i+1)
    # Feed E&E counters
    cause = 'Crossover + Mutations + Normalisation'
    iter = i+1
    feed_ee_counters_logged(deepcopy(childs), cause, iter)
    
    
    # EVALUATIONS - ITERATION END
    
    # Population for next iteration will be constituted of the best vectors in reunited parents and childs.
    pop = []
    pop = deepcopy(parents) + deepcopy(childs)
    # Sort population by score DESC.
    pop = sorted(deepcopy(pop), key=lambda dct: dct['score'], reverse=True)
    # Keep the best vectors by removing the last ones until population size is respected.
    while len(pop) > population_size:
      pop.pop() # Remove last element from population (weakest one)    
    
    # Strongest vector (v_strong) and associated score (max_score)
    over_max_score = False
    if pop[0]['score'] > max_score:
      over_max_score = True
      max_score = pop[0]['score']
      v_strong = deepcopy(pop[0]['vector'])

    iteration_score_evolution.append(pop[0]['score'])
    
    mean_score_pop = 0
    for k in range(len(pop)):
      mean_score_pop += pop[k]['score']
    mean_score_pop /= len(pop)
    
    each_generation_mean_score.append(mean_score_pop)
    
    if over_max_score is True:
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:
      # Convergence Management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score
    
    max_score_evolution.append(max_score)

  if convergence_score == 0: # if convergence not yet reached, convergence is considered to be max_score.
    convergence_iteration = max_score_iteration
    convergence_score = max_score
    
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score, 
            res
          ]




# Genetic Dynashape
# Classic Genetic Algorithm with reshape/resample operations.
def genetic_dynashape():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  global df

  pop = []
  population_size = 4
  
  # Strongest vector
  v_strong = []
  # Score of the strongest vector
  max_score = 0
  
  iteration_score_evolution = []
  each_generation_mean_score = []
  max_score_evolution = []
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0
    
  
  # INITIALISATION
  # Each vector contains as many digits as pairs action-resource in df.
  for k in range(population_size):
    vector = generate_vector()
    vector_score = score(vector)
    pop.append({
                  'vector':vector,
                  'score':vector_score
              })
    # Add in top_vectors and all_vectors.
    add_to_top_vectors(deepcopy(vector), vector_score)   
    all_vectors.append(deepcopy(vector))
    # Transfer newly discovered digits indexes from "ndd" to "dd".
    transfer_from_ndd_to_dd(deepcopy(vector))


  # -------------------------
  # DYNACOUNTERS
  # -------------------------
  # Array of the total number of times a digit was a 1-digit on all iterations.
  dyna_counters = []
  for k in range(len(df)):
    dyna_counters.append({
                                'df_index':k,
                                'counter':0
                            })
  # -------------------------
  

  generation_value = algorithm_iterations
  for i in range(generation_value): # x iterations
    
    print("Iteration N° " + str(i+1) + " ", end="\r")


    # -------------------------
    # Dynamic Reshape
    # -------------------------
    
    # Update dyna_counters.
    # Sort dyna_counters by df_index ASC.
    dyna_counters = sorted(deepcopy(dyna_counters), key=lambda dct: dct['df_index'])
    # Update dyna_counters.
    for k in range(len(pop)):
      for d in range(len(pop[k]['vector'])):
        if pop[k]['vector'][d] == 1:
          dyna_counters[d]['counter'] += 1
        
    # If the mean of dyna_counters counters is superior to a certain amount:
    # - we create a new df by keeping the most x% used digits.
    # - we generate brand new vectors.
    # - we reset dyna_counters.
    
    # Calculate the mean of dyna_counters counters.
    sum_counters = 0
    for k in range(len(dyna_counters)):  
      sum_counters += dyna_counters[k]['counter']
    mean_counters = sum_counters / len(dyna_counters)
    
    if mean_counters > 10 and len(df) > 50:
    
      print("Iteration N° " + str(i+1))
      ic(len(df))
    
      # Sort dyna_counters by counter DESC.
      dyna_counters = sorted(deepcopy(dyna_counters), key=lambda dct: dct['counter'], reverse=True)
      
      # Create a new df.
      new_df = []
      keep_quantity = int(len(df)*0.9)
      for k in range(keep_quantity):
        index_to_keep = dyna_counters[k]['df_index']
        new_df.append(df[index_to_keep])
      df = []
      df = deepcopy(new_df)
        
      # Generate brand new vectors.
      pop = []
      for k in range(population_size):
        vector = generate_vector()
        vector_score = score(vector)
        pop.append({
                      'vector':vector,
                      'score':vector_score
                  })
        # Add in top_vectors and all_vectors.
        add_to_top_vectors(deepcopy(vector), vector_score)   
        all_vectors.append(deepcopy(vector))
        # Transfer newly discovered digits indexes from "ndd" to "dd".
        transfer_from_ndd_to_dd(deepcopy(vector))        
      
      # Reset dyna_counters.
      dyna_counters = []
      for k in range(len(df)):
        dyna_counters.append({
                                    'df_index':k,
                                    'counter':0
                                })

    # -------------------------




    # PARENTS SELECTION + REPRODUCTION
    # Number of couples to create = population_size/2
    # Take 2 parents randomly from the population and make them reproduce.
    # Two parents make two children.
    # To perform a reproduction, we take the left part of the considered parent (until a random hybridation index). We stick the right part of the other parent (from the hybridation index) to the considered parent.
    parents = []
    childs = []
    number_of_couples = int(population_size/2)
    for k in range(number_of_couples):
      # 1st parent
      index = random.randint(0, len(pop)-1)
      parent_1 = pop[index]
      pop.pop(index)
      parents.append(parent_1)
      # 2nd parent
      index = random.randint(0, len(pop)-1)
      parent_2 = pop[index]
      pop.pop(index)
      parents.append(parent_2)
      # 1st child
      child_1 = deepcopy(parent_1)
      hybridation_index = random.randint(0, len(df)-1)
      for j in range(hybridation_index, len(df)):
        child_1['vector'][j] = parent_2['vector'][j]
      childs.append(child_1)
      # 2nd child
      child_2 = deepcopy(parent_2)
      hybridation_index = random.randint(0, len(df)-1)
      for j in range(hybridation_index, len(df)):
        child_2['vector'][j] = parent_1['vector'][j]
      childs.append(child_2)

    # MUTATIONS
    for k in range(len(childs)):
      # While childs[k]['vector'] respects max_value constraints...
      while constraints_max_values_check(childs[k]['vector']) is True:
        # ...Randomly transform a 0-digit into a 1-digit in childs[k]['vector'].
        rand = random.randint(0, len(df)-1)
        childs[k]['vector'][rand] = 1
      # NORMALISATION
      childs[k]['vector'] = normalise(deepcopy(childs[k]['vector'])) 
    
    
    # LAST TREATMENTS
    # Feed E&E counters
    feed_ee_counters(deepcopy(childs))
    # Other last treatments
    for k in range(len(childs)):
      # Update score
      childs[k]['score'] = score(deepcopy(childs[k]['vector']))
      # Add to top_vectors and all_vectors.
      add_to_top_vectors(deepcopy(childs[k]['vector']), deepcopy(childs[k]['score']))
      all_vectors.append(deepcopy(childs[k]['vector']))

    
    # EVALUATIONS - ITERATION END
    
    # Population for next iteration will be constituted of the best vectors in reunited parents and childs.
    pop = []
    pop = deepcopy(parents) + deepcopy(childs)
    # Sort population by score DESC.
    pop = sorted(deepcopy(pop), key=lambda dct: dct['score'], reverse=True)
    # Keep the best vectors by removing the last ones until population size is respected.
    while len(pop) > population_size:
      pop.pop() # Remove last element from population (weakest one)    
    
    # Strongest vector (v_strong) and associated score (max_score)
    over_max_score = False
    if pop[0]['score'] > max_score:
      over_max_score = True
      max_score = pop[0]['score']
      v_strong = deepcopy(pop[0]['vector'])

    iteration_score_evolution.append(pop[0]['score'])
    
    mean_score_pop = 0
    for k in range(len(pop)):
      mean_score_pop += pop[k]['score']
    mean_score_pop /= len(pop)
    
    each_generation_mean_score.append(mean_score_pop)
    
    if over_max_score is True:
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:
      # Convergence Management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score
    
    max_score_evolution.append(max_score)

  if convergence_score == 0: # if convergence not yet reached, convergence is considered to be max_score.
    convergence_iteration = max_score_iteration
    convergence_score = max_score
    
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score, 
            res
          ]




# Spycle
# Spycle = Speed + Cycle
# General Principle :
# Start with a population of vectors containing 1-digits randomly assigned.
# Define the initial speed and the speed_function used for a cycle.
# ITERATION : 
  # Increase speed with the chosen speed_function parameter.
  # Change speed sign. This allows to get a direction alternated between right and left.  
  # For each vector in the population do :
    # Consider all 1-digits in vector and moves them to the right (positive speed) or to the left (negative speed), using the speed value. If there is already a 1-digit where we move, it is a collision and we do nothing (the considered 1-digit is not moved). If there is a 0-digit where we move, the considered 1-digit become a 0-digit and the 0-digit where we move become a 1-digit. We iterate through a copy of vector (ref_vector) in order to not encounter new 1-digits inserted in vector through iterations. 
    # Add a 1-digit to the left or right of each 1-digit in vector. Add to the right if speed is positive and to the left if speed is negative. We iterate through a copy of vector (ref_vector) in order to not encounter new 1-digits inserted in vector through iterations.
    # Normalise the vector.
    # Merge current vector with the strongest_vector by putting each 1-digit of strongest_vector in current vector.
    # Normalise the vector.
    # Add to top_vectors and all_vectors.
  # Replace the best vector if needed.
  # Create new population with the best vectors from past and new population for next iteration.
# NOTA : 
# The chosen speed_function has an impact on results.
# The expansion has an increasing speed and a variable direction alternating between right and left.
# This algorithm represents the fact that the universe or a cellular system is prone to expand and contract following cycles with variable speeds. These cycles are sometimes the theater of collisions like between planets or cellular systems.
# The will behind this process is to guarantee an exploration structured around a function and a variable speed while still being able to exploit the search space sufficiently. This algorithm is highly customisable because changing the speed function can produce radically different results.
def spycle():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  
  pop = []
  population_size = 2

  strongest_vector = []
  max_score = 0
  iteration_score_evolution = []
  max_score_evolution = []
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0

  # PARAMETERS
  speed = 1 # Initial speed.
  speed_function = "+1" # Can also be other functions. See below.
  
  # INITIALISATION
  # The initial population of vectors containing 1-digits randomly determined.
  for k in range(population_size):
    vector = generate_vector()
    vector_score = score(vector)
    pop.append({
                  'vector':vector,
                  'score':vector_score
              })
    # Add to top_vectors and all_vectors.   
    add_to_top_vectors(deepcopy(vector), vector_score)   
    all_vectors.append(deepcopy(vector))
    # Transfer newly discovered digits indexes from "ndd" to "dd".
    transfer_from_ndd_to_dd(deepcopy(vector))
    
  
  # ITERATIONS
  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations

    print("Iteration N° " + str(i+1) + " ", end="\r")
    
    # Memory of population
    pop_memory = deepcopy(pop)
    
    # Increase speed with the chosen speed_function parameter.
    # Add here more function if you want.
    if speed_function == "x2": 
      speed = 2*speed
    elif speed_function == "square":
      speed = speed * speed
    elif speed_function == "x2_+1":
      speed = (2*speed)+1
    elif speed_function == "x2_+2":
      speed = (2*speed)+2
    elif speed_function == "x2_+3":
      speed = (2*speed)+3
    elif speed_function == "+1":
      speed = speed+1
    elif speed_function == "+2":
      speed = speed+2
    elif speed_function == "+_random":
      speed = speed+random.randint(1, 100)
    elif speed_function == "*_random":
      speed = speed*random.randint(1, 100)
    elif speed_function == "/_random":
      speed = int(speed/random.randint(1, 100))
      
    # Changing speed sign.
    # This allows to get a direction alternated between right and left.
    speed *= -1    
    
    # For each vector in the population.
    for p in range(len(pop)):
      
      vector = pop[p]['vector']

      # Each iteration consider all 1-digits and moves them to the right (positive speed) or to the left (negative speed).
      # We iterate through a copy of vector (ref_vector) in order to not encounter new 1-digits inserted in vector through iterations.
      ref_vector = deepcopy(vector)
      for k in range(len(ref_vector)):
        if ref_vector[k] == 1:

          # Set the 1-digit to 0.
          vector[k] = 0
          
          # Set the index where we want to move the 1-digit.
          move_index = k + speed                 

          # Put move_index back in the range of vector if needed.
          move_index %= len(vector)
          
          # Ensure that at move_index we have a 0-digit. Otherwise, there is a collision and we keep vector[k] = 1.
          if vector[abs(move_index)] == 1:
            move_index = k
          
          """
          # Ensure that at move_index we have a 0-digit. Otherwise, there is a collision and we iterate x times, move_index becoming a random index in the range of vector until we find a 0-digit. If we don't find one, we keep vector[k] = 1.
          if vector[abs(move_index)] == 1:
            collision = True
            loop_counter = len(vector)
            while collision is True: 
              if loop_counter == 0:
                collision = False
                move_index = k
              else:
                move_index = random.randint(0, len(vector)-1)
                if vector[move_index] == 0:
                  collision = False
              loop_counter -= 1
          """
          
          # Transform the 0-digit at move_index into a 1-digit
          vector[move_index] = 1
      
      # Add a 1-digit to the left or right of each 1-digit in vector.
      # Add to the right if speed is positive and to the left if speed is negative.
      # We iterate through a copy of vector (ref_vector) in order to not encounter new 1-digits inserted in vector through iterations.
      ref_vector = deepcopy(vector)
      for d in range(len(ref_vector)):
        if ref_vector[d] == 1:
          target_index = 0
          if speed > 0:
            target_index = d+1
          else:
            target_index = d-1
          target_index %= len(vector) # Put target_index back in the range of vector if needed.
          vector[target_index] = 1
      
      # Normalise
      vector = normalise(deepcopy(vector))
      
      # Merge current vector with the strongest_vector by putting each 1-digit of strongest_vector in current vector.
      for d in range(len(strongest_vector)):
        if strongest_vector[d] == 1:
          vector[d] = 1

      # Normalise
      vector = normalise(deepcopy(vector))

      # Update pop
      pop[p]['vector'] = vector


    # --------------------
    # GUIDE E&E.
    if ee_guided is True:
      # Get the vectors.
      vectors = []
      for k in range(len(pop)):
        vectors.append(deepcopy(pop[k]['vector']))
      # Guide.
      vectors = guide_explo(deepcopy(vectors))
      # Update population vectors.
      for k in range(len(vectors)):
        pop[k]['vector'] = deepcopy(vectors[k])
    # --------------------

    # Feed E&E counters
    feed_ee_counters(deepcopy(pop))

    # LAST TREATMENTS
    for k in range(len(pop)):  
      # Update score
      vector_score = score(pop[k]['vector'])
      pop[k]['score'] = vector_score
      # Add to top_vectors and all_vectors.    
      add_to_top_vectors(deepcopy(pop[k]['vector']), vector_score)   
      all_vectors.append(deepcopy(pop[k]['vector']))
      

    # CREATE POPULATION FOR NEXT ITERATION
    # Merge pop and pop_memory.
    pop += pop_memory
    # Sort population by score DESC.
    pop = sorted(deepcopy(pop), key=lambda dct: dct['score'], reverse=True)
    # Keep the best vectors by removing half of pop length.
    for k in range(population_size):
      pop.pop() # Remove last element from population (weakest one)


    # EVALUATIONS - ITERATION END   
    iteration_score = pop[0]['score']
    iteration_score_evolution.append(iteration_score)
    
    if iteration_score > max_score:
      max_score = iteration_score
      strongest_vector = deepcopy(pop[0]['vector'])
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else: 
      # Convergence Management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score

    max_score_evolution.append(max_score)

  if convergence_score == 0: # if convergence not reached, convergence is considered to be the max_score.
      convergence_iteration = max_score_iteration
      convergence_score = max_score
      
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score, 
            res
          ]




def spycle_ee_logged():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  
  pop = []
  population_size = 2

  strongest_vector = []
  max_score = 0
  iteration_score_evolution = []
  max_score_evolution = []
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0

  # PARAMETERS
  speed = 1 # Initial speed.
  speed_function = "+1" # Can also be other functions. See below.
  
  # INITIALISATION
  # Create the log structure for the iteration (here it is iteration number 0 because we are initializing the population).
  ee_logs.append({
                'iteration':0,
                'vectors':[],
                'vectors_quantity':0,
                'top_quantity':0,
                'top_vectors_score':0,
                'all_vectors_coverage':0,
                'top_vectors_constraints_completion':0,
                'similarity_with_top_vectors':0,
                'power':0,
                'hvv':0,
                'hvo':0,
                'd%v':[],
                'd%o':[],
                'hvcv':0,
                'hvco':0
              })
  # The initial population of vectors containing 1-digits randomly determined.
  for k in range(population_size):
    vector = generate_vector()
    vector_score = score(vector)
    pop.append({
                  'vector':vector,
                  'score':vector_score
              })
    # Add to top_vectors and all_vectors.   
    add_to_top_vectors(deepcopy(vector), vector_score)   
    all_vectors.append(deepcopy(vector))

  # Feed ee_counters and the other attributes of the iteration log.
  feed_ee_counters_logged(deepcopy(pop), 'Pure Random Function', 0)
    
  
  # ITERATIONS
  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations

    print("Iteration N° " + str(i+1) + " ", end="\r")
    
    # Memory of population
    pop_memory = deepcopy(pop)
    
    # Increase speed with the chosen speed_function parameter.
    # Add here more function if you want.
    if speed_function == "x2": 
      speed = 2*speed
    elif speed_function == "square":
      speed = speed * speed
    elif speed_function == "x2_+1":
      speed = (2*speed)+1
    elif speed_function == "x2_+2":
      speed = (2*speed)+2
    elif speed_function == "x2_+3":
      speed = (2*speed)+3
    elif speed_function == "+1":
      speed = speed+1
    elif speed_function == "+2":
      speed = speed+2
    elif speed_function == "+_random":
      speed = speed+random.randint(1, 100)
    elif speed_function == "*_random":
      speed = speed*random.randint(1, 100)
    elif speed_function == "/_random":
      speed = int(speed/random.randint(1, 100))
      
    # Changing speed sign.
    # This allows to get a direction alternated between right and left.
    speed *= -1    
    
    # For each vector in the population.
    for p in range(len(pop)):
      
      vector = pop[p]['vector']

      # Each iteration consider all 1-digits and moves them to the right (positive speed) or to the left (negative speed).
      # We iterate through a copy of vector (ref_vector) in order to not encounter new 1-digits inserted in vector through iterations.
      ref_vector = deepcopy(vector)
      for k in range(len(ref_vector)):
        if ref_vector[k] == 1:

          # Set the 1-digit to 0.
          vector[k] = 0
          
          # Set the index where we want to move the 1-digit.
          move_index = k + speed                 

          # Put move_index back in the range of vector if needed.
          move_index %= len(vector)
          
          # Ensure that at move_index we have a 0-digit. Otherwise, there is a collision and we keep vector[k] = 1.
          if vector[abs(move_index)] == 1:
            move_index = k
          
          """
          # Ensure that at move_index we have a 0-digit. Otherwise, there is a collision and we iterate x times, move_index becoming a random index in the range of vector until we find a 0-digit. If we don't find one, we keep vector[k] = 1.
          if vector[abs(move_index)] == 1:
            collision = True
            loop_counter = len(vector)
            while collision is True: 
              if loop_counter == 0:
                collision = False
                move_index = k
              else:
                move_index = random.randint(0, len(vector)-1)
                if vector[move_index] == 0:
                  collision = False
              loop_counter -= 1
          """
          
          # Transform the 0-digit at move_index into a 1-digit
          vector[move_index] = 1
      
      # Add a 1-digit to the left or right of each 1-digit in vector.
      # Add to the right if speed is positive and to the left if speed is negative.
      # We iterate through a copy of vector (ref_vector) in order to not encounter new 1-digits inserted in vector through iterations.
      ref_vector = deepcopy(vector)
      for d in range(len(ref_vector)):
        if ref_vector[d] == 1:
          target_index = 0
          if speed > 0:
            target_index = d+1
          else:
            target_index = d-1
          target_index %= len(vector) # Put target_index back in the range of vector if needed.
          vector[target_index] = 1
      
      # Normalise
      vector = normalise(deepcopy(vector))
      
      # Merge current vector with the strongest_vector by putting each 1-digit of strongest_vector in current vector.
      for d in range(len(strongest_vector)):
        if strongest_vector[d] == 1:
          vector[d] = 1

      # Normalise
      vector = normalise(deepcopy(vector))

      # Update pop
      pop[p]['vector'] = vector


    # --------------------
    # GUIDE E&E.
    if ee_guided is True:
      # Get the vectors.
      vectors = []
      for k in range(len(pop)):
        vectors.append(deepcopy(pop[k]['vector']))
      # Guide.
      vectors = guide_explo(deepcopy(vectors))
      # Update population vectors.
      for k in range(len(vectors)):
        pop[k]['vector'] = deepcopy(vectors[k])
    # --------------------



    # LAST TREATMENTS
    # Create the log structure for the iteration.
    ee_logs.append({
                  'iteration':i+1,
                  'vectors':[],
                  'vectors_quantity':0,
                  'top_quantity':0,
                  'top_vectors_score':0,
                  'all_vectors_coverage':0,
                  'top_vectors_constraints_completion':0,
                  'similarity_with_top_vectors':0,
                  'power':0,
                  'hvv':0,
                  'hvo':0,
                  'd%v':[],
                  'd%o':[],
                  'hvcv':0,
                  'hvco':0
                })
    for k in range(len(pop)):  
      # Update score
      vector_score = score(pop[k]['vector'])
      pop[k]['score'] = vector_score
      # Add to top_vectors and all_vectors.    
      add_to_top_vectors(deepcopy(pop[k]['vector']), vector_score)   
      all_vectors.append(deepcopy(pop[k]['vector']))
    
    # Feed E&E counters
    cause = 'Variations + Normalisation'
    iter = i+1
    feed_ee_counters_logged(deepcopy(pop), cause, iter)    

    # CREATE POPULATION FOR NEXT ITERATION
    # Merge pop and pop_memory.
    pop += pop_memory
    # Sort population by score DESC.
    pop = sorted(deepcopy(pop), key=lambda dct: dct['score'], reverse=True)
    # Keep the best vectors by removing half of pop length.
    for k in range(population_size):
      pop.pop() # Remove last element from population (weakest one)


    # EVALUATIONS - ITERATION END   
    iteration_score = pop[0]['score']
    iteration_score_evolution.append(iteration_score)
    
    if iteration_score > max_score:
      max_score = iteration_score
      strongest_vector = deepcopy(pop[0]['vector'])
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else: 
      # Convergence Management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score

    max_score_evolution.append(max_score)

  if convergence_score == 0: # if convergence not reached, convergence is considered to be the max_score.
      convergence_iteration = max_score_iteration
      convergence_score = max_score
      
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score, 
            res
          ]




# Struggle
# General Principle : 
# Start from a population of vectors which 1-digits are determined randomly.
# ITERATION: 
  # For each vector in the population do :
    # Select a random range from this vector.
    # The two best 0-digit on the range, regarding score, become 1-digit.    
    # The two worse 0-digit on the range, regarding constraints_score, become 1-digit.
    # The worse 1-digit on the range, regarding score, become 0-digit.
    # The best 1-digit on the range, regarding constraints_score, become 0-digit.
    # Normalise the vector. 
    # Add to top_vectors and all_vectors.
  # Replace best vector if needed.
  # Create a population with the best vectors from past and new population for next iteration.
# NOTA : With this process, targeted mutations are applied (small ranges cases) and extended mutations too (large ranges cases), both of them on various random regions. This represents the fact that a wound is absorbed/healed, with possible consequences elsewhere in the organism, and that the organism reinforce the affected area by changing its physiological configuration.
# The will behind this process is to operate exploitation variations conditioned by an exploration decision. Indeed, the determination of the range in terms of place and size is varying which guarantees a good exploration behaviour. We must precise here that this behaviour can sometimes be exploitative but we can assume that most of the time it will not and that when it is, it will bring some interesting variations too.
def struggle():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  
  pop = []
  population_size = 2
  strongest_vector = []
  max_score = 0
  iteration_score_evolution = []
  max_score_evolution = []
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0


  # INITIALISATION
  # The initial population.
  for k in range(population_size):
    vector = generate_vector()
    vector_score = score(vector)
    pop.append({
                  'vector':vector,
                  'score':vector_score
              })
    # Add to top_vectors and all_vectors.
    add_to_top_vectors(deepcopy(vector), vector_score)    
    all_vectors.append(deepcopy(vector))
    # Transfer newly discovered digits indexes from "ndd" to "dd".
    transfer_from_ndd_to_dd(deepcopy(vector)) 

  iteration_number = algorithm_iterations
  for i in range(iteration_number):
    
    print("Iteration N° " + str(i+1) + " ", end="\r")
    
    # Memory of population
    pop_memory = deepcopy(pop)
    
    # For each vector in the population.
    for p in range(len(pop)):
      vector = pop[p]['vector']
      
      # Determination of the random range.
      indice_1 = 0
      indice_2 = 0
      while indice_1 == indice_2:
        indice_1 = random.randint(0, len(df)-1)
        indice_2 = random.randint(0, len(df)-1)
      plage_start = 0
      plage_end = 0
      if indice_1 > indice_2:
        plage_start = indice_2
        plage_end = indice_1
      else:
        plage_start = indice_1
        plage_end = indice_2

      
      # DEALING WITH 0-DIGITS ON THE RANGE
      # Build an array containing each 0-digit index on the range with their associated scores and constraints_scores.
      indexes_scores = []
      for k in range(plage_start, plage_end+1):
        if vector[k] == 0:
          indexes_scores.append({
                                  'index':k, 
                                  'score':resource_score(k),
                                  'constraints_score':resource_constraints_score(k)
                                })    

      if len(indexes_scores) > 0: 
        # The two best 0-digit on the range, regarding score, become 1-digit.
        # Sort indexes_scores by score DESC.  
        indexes_scores = sorted(deepcopy(indexes_scores), key=lambda dct: dct['score'], reverse=True)
        vector[indexes_scores[0]['index']] = 1
        if len(indexes_scores) > 1:
          vector[indexes_scores[1]['index']] = 1
        
        # The two worse 0-digit on the range, regarding constraints_score, become 1-digit.
        # Sort indexes_scores by constraints_score ASC.  
        indexes_scores = sorted(deepcopy(indexes_scores), key=lambda dct: dct['constraints_score'])
        vector[indexes_scores[0]['index']] = 1
        if len(indexes_scores) > 1:
          vector[indexes_scores[1]['index']] = 1


      # DEALING WITH 1-DIGITS ON THE RANGE
      # Build an array containing each 1-digit index on the range with their associated scores and constraints_scores.
      indexes_scores = []
      for k in range(plage_start, plage_end+1):
        if vector[k] == 1:
          indexes_scores.append({
                                  'index':k, 
                                  'score':resource_score(k),
                                  'constraints_score':resource_constraints_score(k)
                                })    
      
      if len(indexes_scores) > 0: 
        # The worse 1-digit on the range, regarding score, become 0-digit.
        # Sort indexes_scores by score ASC.  
        indexes_scores = sorted(deepcopy(indexes_scores), key=lambda dct: dct['score'])
        vector[indexes_scores[0]['index']] = 0
        
        # The best 1-digit on the range, regarding constraints_score, become 0-digit.
        # Sort indexes_scores by constraints_score DESC.  
        indexes_scores = sorted(deepcopy(indexes_scores), key=lambda dct: dct['constraints_score'], reverse=True)
        vector[indexes_scores[0]['index']] = 0

      # Normalise
      vector = normalise(deepcopy(vector))
      
      # Update population
      pop[p]['vector'] = deepcopy(vector)      


    # --------------------
    # GUIDE E&E.
    if ee_guided is True:
      # Get the vectors.
      vectors = []
      for k in range(len(pop)):
        vectors.append(deepcopy(pop[k]['vector']))
      # Guide.
      vectors = guide_explo(deepcopy(vectors))
      # Update population vectors and scores.
      for k in range(len(vectors)):
        pop[k]['vector'] = deepcopy(vectors[k])
    # --------------------

    # Feed E&E counters
    feed_ee_counters(deepcopy(pop))

    # LAST TREATMENTS
    for k in range(len(pop)):  
      # Update score
      vector_score = score(pop[k]['vector'])
      pop[k]['score'] = vector_score
      # Add to top_vectors and all_vectors.    
      add_to_top_vectors(deepcopy(pop[k]['vector']), vector_score)   
      all_vectors.append(deepcopy(pop[k]['vector']))


    # CREATE POPULATION FOR NEXT ITERATION
    # Merge pop and pop_memory.
    pop += pop_memory
    # Sort population by score DESC.
    pop = sorted(deepcopy(pop), key=lambda dct: dct['score'], reverse=True)
    # Keep the best vectors by removing half of pop length.
    for k in range(population_size):
      pop.pop() # Remove last element from population (weakest one)
      
      
    # EVALUATIONS - ITERATION END 
    iteration_score = pop[0]['score']
    iteration_score_evolution.append(iteration_score)
    
    if iteration_score > max_score:
      max_score = iteration_score
      strongest_vector = pop[0]['vector']
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:   
      # Convergence Management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score

    max_score_evolution.append(max_score)

  if convergence_score == 0: # if convergence not reached, convergence is considered to be the max_score.
    convergence_iteration = max_score_iteration
    convergence_score = max_score
    
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score, 
            res
          ]




def struggle_ee_logged():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  
  pop = []
  population_size = 2
  strongest_vector = []
  max_score = 0
  iteration_score_evolution = []
  max_score_evolution = []
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0


  # INITIALISATION
  # Create the log structure for the iteration (here it is iteration number 0 because we are initializing the population).
  ee_logs.append({
                'iteration':0,
                'vectors':[],
                'vectors_quantity':0,
                'top_quantity':0,
                'top_vectors_score':0,
                'all_vectors_coverage':0,
                'top_vectors_constraints_completion':0,
                'similarity_with_top_vectors':0,
                'power':0,
                'hvv':0,
                'hvo':0,
                'd%v':[],
                'd%o':[],
                'hvcv':0,
                'hvco':0
              })
  # The initial population.
  for k in range(population_size):
    vector = generate_vector()
    vector_score = score(vector)
    pop.append({
                  'vector':vector,
                  'score':vector_score
              })
    # Add to top_vectors and all_vectors.
    add_to_top_vectors(deepcopy(vector), vector_score)    
    all_vectors.append(deepcopy(vector))
  
  # Feed ee_counters and the other attributes of the iteration log.
  feed_ee_counters_logged(deepcopy(pop), 'Pure Random Function', 0) 

  iteration_number = algorithm_iterations
  for i in range(iteration_number):
    
    print("Iteration N° " + str(i+1) + " ", end="\r")
    
    # Memory of population
    pop_memory = deepcopy(pop)
    
    # For each vector in the population.
    for p in range(len(pop)):
      vector = pop[p]['vector']
      
      # Determination of the random range.
      indice_1 = 0
      indice_2 = 0
      while indice_1 == indice_2:
        indice_1 = random.randint(0, len(df)-1)
        indice_2 = random.randint(0, len(df)-1)
      plage_start = 0
      plage_end = 0
      if indice_1 > indice_2:
        plage_start = indice_2
        plage_end = indice_1
      else:
        plage_start = indice_1
        plage_end = indice_2

      
      # DEALING WITH 0-DIGITS ON THE RANGE
      # Build an array containing each 0-digit index on the range with their associated scores and constraints_scores.
      indexes_scores = []
      for k in range(plage_start, plage_end+1):
        if vector[k] == 0:
          indexes_scores.append({
                                  'index':k, 
                                  'score':resource_score(k),
                                  'constraints_score':resource_constraints_score(k)
                                })    

      if len(indexes_scores) > 0: 
        # The two best 0-digit on the range, regarding score, become 1-digit.
        # Sort indexes_scores by score DESC.  
        indexes_scores = sorted(deepcopy(indexes_scores), key=lambda dct: dct['score'], reverse=True)
        vector[indexes_scores[0]['index']] = 1
        if len(indexes_scores) > 1:
          vector[indexes_scores[1]['index']] = 1
        
        # The two worse 0-digit on the range, regarding constraints_score, become 1-digit.
        # Sort indexes_scores by constraints_score ASC.  
        indexes_scores = sorted(deepcopy(indexes_scores), key=lambda dct: dct['constraints_score'])
        vector[indexes_scores[0]['index']] = 1
        if len(indexes_scores) > 1:
          vector[indexes_scores[1]['index']] = 1


      # DEALING WITH 1-DIGITS ON THE RANGE
      # Build an array containing each 1-digit index on the range with their associated scores and constraints_scores.
      indexes_scores = []
      for k in range(plage_start, plage_end+1):
        if vector[k] == 1:
          indexes_scores.append({
                                  'index':k, 
                                  'score':resource_score(k),
                                  'constraints_score':resource_constraints_score(k)
                                })    
      
      if len(indexes_scores) > 0: 
        # The worse 1-digit on the range, regarding score, become 0-digit.
        # Sort indexes_scores by score ASC.  
        indexes_scores = sorted(deepcopy(indexes_scores), key=lambda dct: dct['score'])
        vector[indexes_scores[0]['index']] = 0
        
        # The best 1-digit on the range, regarding constraints_score, become 0-digit.
        # Sort indexes_scores by constraints_score DESC.  
        indexes_scores = sorted(deepcopy(indexes_scores), key=lambda dct: dct['constraints_score'], reverse=True)
        vector[indexes_scores[0]['index']] = 0

      # Normalise
      vector = normalise(deepcopy(vector))
      
      # Update population
      pop[p]['vector'] = deepcopy(vector)      


    # --------------------
    # GUIDE E&E.
    if ee_guided is True:
      # Get the vectors.
      vectors = []
      for k in range(len(pop)):
        vectors.append(deepcopy(pop[k]['vector']))
      # Guide.
      vectors = guide_explo(deepcopy(vectors))
      # Update population vectors and scores.
      for k in range(len(vectors)):
        pop[k]['vector'] = deepcopy(vectors[k])
    # --------------------


    # LAST TREATMENTS
    # Create the log structure for the iteration.
    ee_logs.append({
                  'iteration':i+1,
                  'vectors':[],
                  'vectors_quantity':0,
                  'top_quantity':0,
                  'top_vectors_score':0,
                  'all_vectors_coverage':0,
                  'top_vectors_constraints_completion':0,
                  'similarity_with_top_vectors':0,
                  'power':0,
                  'hvv':0,
                  'hvo':0,
                  'd%v':[],
                  'd%o':[],
                  'hvcv':0,
                  'hvco':0
                })
    for k in range(len(pop)):  
      # Update score
      vector_score = score(pop[k]['vector'])
      pop[k]['score'] = vector_score
      # Add to top_vectors and all_vectors.    
      add_to_top_vectors(deepcopy(pop[k]['vector']), vector_score)   
      all_vectors.append(deepcopy(pop[k]['vector']))

    # Feed E&E counters
    cause = 'Variations + Normalisation'
    iter = i+1
    feed_ee_counters_logged(deepcopy(pop), cause, iter)
    

    # CREATE POPULATION FOR NEXT ITERATION
    # Merge pop and pop_memory.
    pop += pop_memory
    # Sort population by score DESC.
    pop = sorted(deepcopy(pop), key=lambda dct: dct['score'], reverse=True)
    # Keep the best vectors by removing half of pop length.
    for k in range(population_size):
      pop.pop() # Remove last element from population (weakest one)
      
      
    # EVALUATIONS - ITERATION END 
    iteration_score = pop[0]['score']
    iteration_score_evolution.append(iteration_score)
    
    if iteration_score > max_score:
      max_score = iteration_score
      strongest_vector = pop[0]['vector']
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:   
      # Convergence Management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score

    max_score_evolution.append(max_score)

  if convergence_score == 0: # if convergence not reached, convergence is considered to be the max_score.
    convergence_iteration = max_score_iteration
    convergence_score = max_score
    
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score, 
            res
          ]




# Ramify
# General Principle : 
# Start from a vector full of 1-digits (tree trunk).
# ITERATION:
  # Ramify this vector into two ramifications by cutting it in half (in terms of 1-digits) and by creating its reverse (in terms of 1-digits). 
  # Keep the best of the two (in terms of score).
  # Ramify again until best ramification kept respects max_value constraints.
  # Merge the final ramification with the strongest_ramification known until now (root contributing the most to the tree life), by putting each 1-digit of strongest_ramification in the ramification. This step plays the role of memory.
  # Normalise the final ramification.
  # GUIDE E&E.
  # Merge again the final ramification with the strongest_ramification known until now (root contributing the most to the tree life), by putting each 1-digit of strongest_ramification in the ramification. This step plays the role of memory.
  # Normalise the final ramification.  
  # Add to top_vectors and all_vectors.
  # Replace strongest ramification by the final ramification if it surpasses it regarding scores.
  # Go to next iteration restarting from scratch with a vector full of 1-digits.
# NOTA : It is also possible to return both sub_ramification during the ramify process to have a better score but a different behaviour (it multiplies the execution time by more than 2 but increase the speed to reach the convergence (less iterations needed)). In this case, there is a portion of code at the beginning of the ramify() function that controls the number of ramifications and makes sure that it doesn't exceed a certain amount (in order to avoid an execution time explosion). The more the amount of ramifications is high, the more the execution time explode but the more the score is better too.
# We assume that this algorithm is quite weird at first glance because it doesn’t manage memory in a traditional way and because it restarts from scratch at each new iteration which gives a feeling of memory destruction and of lost information. However, memory is in fact clearly present when we merge the final ramification with the strongest one known until now. Moreover, the approach is also unconventional in the sense that we reach an acceptable vector by considering it full of 1-digits first and by cutting it into pieces.
# The will behind this process is to counterbalance the exploitative part (merge) by performing strong exploration cuts (restart from scratch).
def ramify():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  
  # Function to ramify one step beyond.  
  def ramify(ramifications):
    new_ramifications = []
    
    # Before ramifying, we make sure that there are no more than 2 ramifications.
    # If there are more, we sort ramifications by score DESC and keep the 2 best ones.
    if len(ramifications) > 2:
      best_ones = []
      for k in range(len(ramifications)):
        ramification_score = score(ramifications[k])
        best_ones.append({
                            'vector':ramifications[k],
                            'score':ramification_score
                        })
      # Sort best_ones by score DESC.
      best_ones = sorted(deepcopy(best_ones), key=lambda dct: dct['score'], reverse=True)
      # Remove worst ones from best_ones until its size is 2.
      while len(best_ones) > 2:
        best_ones.pop()
      # Add each ramification from best_ones into a fresh new ramifications.
      ramifications = []
      for k in range(len(best_ones)):
        ramifications.append(best_ones[k]['vector'])
        
    # RAMIFY
    for k in range(len(ramifications)):
      # Count the number of 1-digits in ramifications[k]
      number_1_digits = 0
      for j in range(len(ramifications[k])):
        if ramifications[k][j] == 1:
          number_1_digits += 1
      number_1_digits_to_convert = int(number_1_digits/2)
      
      # Create sub_ramification_1 with the digits of ramifications[k] changing half of the 1-digits into 0-digits randomly.
      sub_ramification_1 = deepcopy(ramifications[k])
      digits_converted = 0
      while digits_converted < number_1_digits_to_convert:
        for j in range(len(ramifications[k])):
          if digits_converted < number_1_digits_to_convert:  
            if ramifications[k][j] == 1:
              new_digit = random.randint(0, 1)
              if new_digit == 0:
                sub_ramification_1[j] = 0
                digits_converted += 1  

      # Create sub_ramification_2 as the reverse of sub_ramification_1 :
      #  - 1-digits passed to 0 in sub_ramification_1 are 1-digits in sub_ramification_2
      #  - 1-digits in sub_ramification_1 are 0-digits in sub_ramification_2. 
      sub_ramification_2 = deepcopy(ramifications[k])
      for j in range(len(ramifications[k])):
        if ramifications[k][j] == 1 and sub_ramification_1[j] == 1:
          sub_ramification_2[j] = 0
        elif ramifications[k][j] == 1 and sub_ramification_1[j] == 0:
          sub_ramification_2[j] = 1
      
      # Feed new_ramifications with both sub_ramification.
      # This multiply the execution time by more than 2 but increase the score and the speed to reach the convergence (less iterations needed).
      new_ramifications.append(sub_ramification_1)
      new_ramifications.append(sub_ramification_2)
      
      """
      # FEED ALTERNATIVE
      # Feed new_ramifications with the best sub_ramification.
      # Doing that implies that, at the end, only one ramification is present in ramifications. As a consequence, mean_score will often be equal to max_score.
      if score(sub_ramification_1) >= score(sub_ramification_2):
        new_ramifications.append(sub_ramification_1)
      else:
        new_ramifications.append(sub_ramification_2)
      """
      
    return new_ramifications


  # Variables Initialisation
  max_score = 0
  mean_score_evolution = []
  iteration_score_evolution = []
  max_score_evolution = []
  ramifications = []

  # The strongest ramification is initialised as a vector containing 1-digits randomly determined.
  strongest_ramification = generate_vector()

  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0

  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations
    
    print("Iteration N° " + str(i+1) + " ", end="\r")

    # BUILD ramifications
    ramifications = []
    # Adding a full of 1-digits ramification into ramifications.
    start_ramification = []
    for k in range(len(df)):
      start_ramification.append(1)
    ramifications.append(start_ramification)
    
    # While the 1st ramification in ramifications does not respect max_value constraints, ramifications is ramified.
    while constraints_max_values_check(ramifications[0]) is False:
      ramifications = ramify(ramifications)

    # Merge each ramification with the strongest_ramification by putting each 1-digit of strongest_ramification in each ramification.
    for k in range(len(ramifications)):
      for d in range(len(strongest_ramification)):
        if strongest_ramification[d] == 1:
          ramifications[k][d] = 1

    # NORMALISE
    for k in range(len(ramifications)):
      ramifications[k] = normalise(deepcopy(ramifications[k]))
      
   
    # --------------------
    # GUIDE E&E.
    if ee_guided is True:
      ramifications = guide_explo(deepcopy(ramifications))
    # --------------------

    
    # Merge again each ramification with the strongest_ramification by putting each 1-digit of strongest_ramification in each ramification.
    for k in range(len(ramifications)):
      for d in range(len(strongest_ramification)):
        if strongest_ramification[d] == 1:
          ramifications[k][d] = 1

    # NORMALISE
    for k in range(len(ramifications)):
      ramifications[k] = normalise(deepcopy(ramifications[k]))

    # Feed E&E counters
    pop = []
    for k in range(len(ramifications)):
      pop.append({
                    'vector':ramifications[k],
                    'score':0
                })
    feed_ee_counters(deepcopy(pop))
      
    # Adding to top_vectors and all_vectors and creating scores array and just_scores array.    
    scores = []
    just_scores = []
    for k in range(len(ramifications)):
      vector_score = score(ramifications[k])
      add_to_top_vectors(deepcopy(ramifications[k]), vector_score)    
      all_vectors.append(deepcopy(ramifications[k]))
      scores.append({
                      'vector':deepcopy(ramifications[k]),
                      'score':vector_score
                    })
      just_scores.append(vector_score)
    # Sort scores by score DESC.
    scores = sorted(deepcopy(scores), key=lambda dct: dct['score'], reverse=True)   

    
    # EVALUATIONS - ITERATION END 
    over_max_score = False
    if scores[0]['score'] > max_score:
      over_max_score = True
      max_score = scores[0]['score']
      strongest_ramification = deepcopy(scores[0]['vector'])
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    
    if over_max_score is False:
      # Convergence Management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score

    iteration_score = scores[0]['score']
    iteration_score_evolution.append(iteration_score)
    
    max_score_evolution.append(max_score)
    
    mean_score_evolution.append(
                                  mean(just_scores)
                                )

  if convergence_score == 0: # if convergence not reached, the convergence is considered to be the max_score.
    convergence_iteration = max_score_iteration
    convergence_score = max_score
    
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score, 
            res
          ]




def ramify_ee_logged():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  
  # Function to ramify one step beyond.  
  def ramify(ramifications):
    new_ramifications = []
    
    # Before ramifying, we make sure that there are no more than 2 ramifications.
    # If there are more, we sort ramifications by score DESC and keep the 2 best ones.
    if len(ramifications) > 2:
      best_ones = []
      for k in range(len(ramifications)):
        ramification_score = score(ramifications[k])
        best_ones.append({
                            'vector':ramifications[k],
                            'score':ramification_score
                        })
      # Sort best_ones by score DESC.
      best_ones = sorted(deepcopy(best_ones), key=lambda dct: dct['score'], reverse=True)
      # Remove worst ones from best_ones until its size is 2.
      while len(best_ones) > 2:
        best_ones.pop()
      # Add each ramification from best_ones into a fresh new ramifications.
      ramifications = []
      for k in range(len(best_ones)):
        ramifications.append(best_ones[k]['vector'])
        
    # RAMIFY
    for k in range(len(ramifications)):
      # Count the number of 1-digits in ramifications[k]
      number_1_digits = 0
      for j in range(len(ramifications[k])):
        if ramifications[k][j] == 1:
          number_1_digits += 1
      number_1_digits_to_convert = int(number_1_digits/2)
      
      # Create sub_ramification_1 with the digits of ramifications[k] changing half of the 1-digits into 0-digits randomly.
      sub_ramification_1 = deepcopy(ramifications[k])
      digits_converted = 0
      while digits_converted < number_1_digits_to_convert:
        for j in range(len(ramifications[k])):
          if digits_converted < number_1_digits_to_convert:  
            if ramifications[k][j] == 1:
              new_digit = random.randint(0, 1)
              if new_digit == 0:
                sub_ramification_1[j] = 0
                digits_converted += 1  

      # Create sub_ramification_2 as the reverse of sub_ramification_1 :
      #  - 1-digits passed to 0 in sub_ramification_1 are 1-digits in sub_ramification_2
      #  - 1-digits in sub_ramification_1 are 0-digits in sub_ramification_2. 
      sub_ramification_2 = deepcopy(ramifications[k])
      for j in range(len(ramifications[k])):
        if ramifications[k][j] == 1 and sub_ramification_1[j] == 1:
          sub_ramification_2[j] = 0
        elif ramifications[k][j] == 1 and sub_ramification_1[j] == 0:
          sub_ramification_2[j] = 1
      
      # Feed new_ramifications with both sub_ramification.
      # This multiply the execution time by more than 2 but increase the score and the speed to reach the convergence (less iterations needed).
      new_ramifications.append(sub_ramification_1)
      new_ramifications.append(sub_ramification_2)
      
      """
      # FEED ALTERNATIVE
      # Feed new_ramifications with the best sub_ramification.
      # Doing that implies that, at the end, only one ramification is present in ramifications. As a consequence, mean_score will often be equal to max_score.
      if score(sub_ramification_1) >= score(sub_ramification_2):
        new_ramifications.append(sub_ramification_1)
      else:
        new_ramifications.append(sub_ramification_2)
      """
      
    return new_ramifications


  # Variables Initialisation
  max_score = 0
  mean_score_evolution = []
  iteration_score_evolution = []
  max_score_evolution = []
  ramifications = []

  # The strongest ramification is initialised as a vector containing 1-digits randomly determined.
  strongest_ramification = generate_vector()

  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0

  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations
    
    print("Iteration N° " + str(i+1) + " ", end="\r")

    # BUILD ramifications
    ramifications = []
    # Adding a full of 1-digits ramification into ramifications.
    start_ramification = []
    for k in range(len(df)):
      start_ramification.append(1)
    ramifications.append(start_ramification)
    
    # While the 1st ramification in ramifications does not respect max_value constraints, ramifications is ramified.
    while constraints_max_values_check(ramifications[0]) is False:
      ramifications = ramify(ramifications)

    # Merge each ramification with the strongest_ramification by putting each 1-digit of strongest_ramification in each ramification.
    for k in range(len(ramifications)):
      for d in range(len(strongest_ramification)):
        if strongest_ramification[d] == 1:
          ramifications[k][d] = 1

    # NORMALISE
    for k in range(len(ramifications)):
      ramifications[k] = normalise(deepcopy(ramifications[k]))
      
   
    # --------------------
    # GUIDE E&E.
    if ee_guided is True:
      ramifications = guide_explo(deepcopy(ramifications))
    # --------------------

    
    # Merge again each ramification with the strongest_ramification by putting each 1-digit of strongest_ramification in each ramification.
    for k in range(len(ramifications)):
      for d in range(len(strongest_ramification)):
        if strongest_ramification[d] == 1:
          ramifications[k][d] = 1

    # NORMALISE
    for k in range(len(ramifications)):
      ramifications[k] = normalise(deepcopy(ramifications[k]))

    

    # LAST TREATMENTS
    # Create the log structure for the iteration.
    ee_logs.append({
                  'iteration':i+1,
                  'vectors':[],
                  'vectors_quantity':0,
                  'top_quantity':0,
                  'top_vectors_score':0,
                  'all_vectors_coverage':0,
                  'top_vectors_constraints_completion':0,
                  'similarity_with_top_vectors':0,
                  'power':0,
                  'hvv':0,
                  'hvo':0,
                  'd%v':[],
                  'd%o':[],
                  'hvcv':0,
                  'hvco':0
                })     
    # Adding to top_vectors and all_vectors and creating scores array and just_scores array.    
    scores = []
    just_scores = []
    for k in range(len(ramifications)):
      vector_score = score(ramifications[k])
      add_to_top_vectors(deepcopy(ramifications[k]), vector_score)    
      all_vectors.append(deepcopy(ramifications[k]))
      scores.append({
                      'vector':deepcopy(ramifications[k]),
                      'score':vector_score
                    })
      just_scores.append(vector_score)
    # Sort scores by score DESC.
    scores = sorted(deepcopy(scores), key=lambda dct: dct['score'], reverse=True)   
    # Feed E&E counters
    pop = []
    for k in range(len(ramifications)):
      pop.append({
                    'vector':ramifications[k],
                    'score':0
                })
    cause = 'Variations + Normalisation'
    iter = i
    feed_ee_counters_logged(deepcopy(pop), cause, iter)
    
    # EVALUATIONS - ITERATION END 
    over_max_score = False
    if scores[0]['score'] > max_score:
      over_max_score = True
      max_score = scores[0]['score']
      strongest_ramification = deepcopy(scores[0]['vector'])
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    
    if over_max_score is False:
      # Convergence Management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score

    iteration_score = scores[0]['score']
    iteration_score_evolution.append(iteration_score)
    
    max_score_evolution.append(max_score)
    
    mean_score_evolution.append(
                                  mean(just_scores)
                                )

  if convergence_score == 0: # if convergence not reached, the convergence is considered to be the max_score.
    convergence_iteration = max_score_iteration
    convergence_score = max_score
    
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score, 
            res
          ]




# Spasm
# General Principle : 
# Start with a population of vectors containing 1-digits randomly determined.
# ITERATION :
  # For each vector do :
    # MACRO SPASM : Select a random range from vector. All digits in it become 1-digits.
    # Normalise the vector.
    # MICRO SPASM ON FULL VECTOR : Select a random index from vector. The digit at index and around him become 1-digits.
    # MICRO SPASM OUTSIDE RANGE USED FOR MACRO SPASM : Select a random index from vector but outside the previous random range selected for the macro spasm. The digit at index and around him become 1-digits.
    # Normalise the vector.
    # Add to top_vectors and all_vectors.
  # Replace best vector if needed.
  # Create new population with best vectors from past and new population for next iteration.
# The approach is quite radical here because we enrich the score by mutating a full range into 1-digits (MACRO SPASM). However, we must put things into perspective as the range can be very small. Moreover, the two MICRO SPASM add some more enrichment but around specific digits.
# The will behind this process is to bring a good amount of chaos (exploration) at the start of each iteration (MACRO SPASM). This chaos can’t destroy good past solutions, because it can only surround them, and allows to discover potential new interesting zones in the search space. We must also precise that these surrounding/discovery motions can also be profitable if they manage to pull out the algorithm from a not so good local optimum. MICRO SPASMs on their side are just here to add some more enrichment pings (small scale ones) which can be both exploratory and exploitative.
def spasm():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  
  pop = []
  population_size = 2
  
  strongest_vector = []
  max_score = 0
  iteration_score_evolution = []
  max_score_evolution = []
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0


  # INITIALISATION
  # The initial population of vectors containing 1-digits randomly determined.
  for k in range(population_size):
    vector = generate_vector()
    vector_score = score(vector)
    pop.append({
                  'vector':vector,
                  'score':vector_score
              })
    # Add to top_vectors and all_vectors.   
    add_to_top_vectors(deepcopy(vector), vector_score)   
    all_vectors.append(deepcopy(vector))
    # Transfer newly discovered digits indexes from "ndd" to "dd".
    transfer_from_ndd_to_dd(deepcopy(vector))
    

  # ITERATIONS
  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations

    print("Iteration N° " + str(i+1) + " ", end="\r")
    
    # Memory of population
    pop_memory = deepcopy(pop)

    for p in range(len(pop)):
      
      vector = pop[p]['vector']

      # MACRO SPASM => ON A FULL RANDOM RANGE
      # Select a random range from vector. All digits in it become 1-digits.
      range_start = random.randint(0, len(vector)-2)
      range_end = random.randint(range_start+1, len(vector)-1)
      for k in range(range_start, range_end+1):
        vector[k] = 1
      
      # Normalise
      vector = normalise(deepcopy(vector))
      
      # MICRO SPASM => AROUND A RANDOM INDEX FROM FULL VECTOR
      # Select a random index from vector. The digit at index and around him become 1-digits.
      random_index = random.randint(1, len(vector)-2)
      vector[random_index-1] = 1
      vector[random_index] = 1
      vector[random_index+1] = 1
      
      """
      # Normalise
      vector = normalise(deepcopy(vector))
      """
      
      # MICRO SPASM => AROUND A RANDOM INDEX OUTSIDE RANGE USED FOR MACRO SPASM
      # Select a random index from vector but outside the previous random range selected for the macro spasm. The digit at index and around him become 1-digits.
      candidate_indexes = []
      for k in range(len(vector)):
        candidate_indexes.append(k)
      for k in range(range_start, range_end+1):
        candidate_indexes.remove(k)
      if len(candidate_indexes) > 0:
        random_index = random.choice(candidate_indexes)
        vector[random_index] = 1
        if random_index == 0:
          vector[len(vector)-1] = 1
          vector[random_index+1] = 1   
        elif random_index == (len(vector)-1):
          vector[random_index-1] = 1
          vector[0] = 1    
        else:
          vector[random_index-1] = 1
          vector[random_index+1] = 1 
      
      # Normalise
      vector = normalise(deepcopy(vector))
      
      # Update population
      pop[p]['vector'] = deepcopy(vector)      


    # --------------------
    # GUIDE E&E.
    if ee_guided is True:
      # Get the vectors.
      vectors = []
      for k in range(len(pop)):
        vectors.append(deepcopy(pop[k]['vector']))
      # Guide.
      vectors = guide_explo(deepcopy(vectors))
      # Update population vectors and scores.
      for k in range(len(vectors)):
        pop[k]['vector'] = deepcopy(vectors[k])
    # --------------------
    """
    # --------------------
    # EXPLO VARIATIONS WITHOUT GUIDING
    # Get the vectors.
    vectors = []
    for k in range(len(pop)):
      vectors.append(deepcopy(pop[k]['vector']))
    # Explo Variations.
    vectors = explo_var(deepcopy(vectors))
    # Update population vectors and scores.
    for k in range(len(vectors)):
      pop[k]['vector'] = deepcopy(vectors[k])
    # --------------------
    """
    
    # Feed E&E counters
    feed_ee_counters(deepcopy(pop))
    
    # LAST TREATMENTS
    for k in range(len(pop)):  
      # Update score
      vector_score = score(pop[k]['vector'])
      pop[k]['score'] = vector_score
      # Add to top_vectors and all_vectors.    
      add_to_top_vectors(deepcopy(pop[k]['vector']), vector_score)   
      all_vectors.append(deepcopy(pop[k]['vector']))


    # CREATE POPULATION FOR NEXT ITERATION
    # Merge pop and pop_memory.
    pop += pop_memory
    # Sort population by score DESC.
    pop = sorted(deepcopy(pop), key=lambda dct: dct['score'], reverse=True)
    # Keep the best vectors by removing half of pop length.
    for k in range(population_size):
      pop.pop() # Remove last element from population (weakest one)
    

    # EVALUATIONS - ITERATION END 
    iteration_score = pop[0]['score']
    iteration_score_evolution.append(iteration_score)
    
    if iteration_score > max_score:
      max_score = iteration_score
      strongest_vector = deepcopy(pop[0]['vector'])
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:
    
      # --------------------
      # GUIDE E&E.
      if ee_guided is True:
        # Get the vectors.
        vectors = []
        for k in range(len(pop)):
          vectors.append(pop[k]['vector'])
        # Guide.
        vectors = guide_explo(vectors)
        # Update population vectors and scores.
        for k in range(len(vectors)):
          pop[k]['vector'] = vectors[k]
          pop[k]['score'] = score(vectors[k])
      # --------------------     
    
      # Convergence Management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score

    max_score_evolution.append(max_score)

  if convergence_score == 0: # if convergence not reached, convergence is considered to be the max_score.
      convergence_iteration = max_score_iteration
      convergence_score = max_score
      
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score, 
            res
          ]




def spasm_ee_logged():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  
  pop = []
  population_size = 2
  
  strongest_vector = []
  max_score = 0
  iteration_score_evolution = []
  max_score_evolution = []
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0


  # INITIALISATION
  # Create the log structure for the iteration (here it is iteration number 0 because we are initializing the population).
  ee_logs.append({
                'iteration':0,
                'vectors':[],
                'vectors_quantity':0,
                'top_quantity':0,
                'top_vectors_score':0,
                'all_vectors_coverage':0,
                'top_vectors_constraints_completion':0,
                'similarity_with_top_vectors':0,
                'power':0,
                'hvv':0,
                'hvo':0,
                'd%v':[],
                'd%o':[],
                'hvcv':0,
                'hvco':0
              })
  # The initial population of vectors containing 1-digits randomly determined.
  for k in range(population_size):
    vector = generate_vector()
    vector_score = score(vector)
    pop.append({
                  'vector':vector,
                  'score':vector_score
              })
    # Add to top_vectors and all_vectors.   
    add_to_top_vectors(deepcopy(vector), vector_score)   
    all_vectors.append(deepcopy(vector))

  # Feed ee_counters and the other attributes of the iteration log.
  feed_ee_counters_logged(deepcopy(pop), 'Pure Random Function', 0)
    

  # ITERATIONS
  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations

    print("Iteration N° " + str(i+1) + " ", end="\r")
    
    # Memory of population
    pop_memory = deepcopy(pop)

    for p in range(len(pop)):
      
      vector = pop[p]['vector']

      # MACRO SPASM => ON A FULL RANDOM RANGE
      # Select a random range from vector. All digits in it become 1-digits.
      range_start = random.randint(0, len(vector)-2)
      range_end = random.randint(range_start+1, len(vector)-1)
      for k in range(range_start, range_end+1):
        vector[k] = 1
      
      # Normalise
      vector = normalise(deepcopy(vector))
      
      # MICRO SPASM => AROUND A RANDOM INDEX FROM FULL VECTOR
      # Select a random index from vector. The digit at index and around him become 1-digits.
      random_index = random.randint(1, len(vector)-2)
      vector[random_index-1] = 1
      vector[random_index] = 1
      vector[random_index+1] = 1
      
      """
      # Normalise
      vector = normalise(deepcopy(vector))
      """
      
      # MICRO SPASM => AROUND A RANDOM INDEX OUTSIDE RANGE USED FOR MACRO SPASM
      # Select a random index from vector but outside the previous random range selected for the macro spasm. The digit at index and around him become 1-digits.
      candidate_indexes = []
      for k in range(len(vector)):
        candidate_indexes.append(k)
      for k in range(range_start, range_end+1):
        candidate_indexes.remove(k)
      if len(candidate_indexes) > 0:
        random_index = random.choice(candidate_indexes)
        vector[random_index] = 1
        if random_index == 0:
          vector[len(vector)-1] = 1
          vector[random_index+1] = 1   
        elif random_index == (len(vector)-1):
          vector[random_index-1] = 1
          vector[0] = 1    
        else:
          vector[random_index-1] = 1
          vector[random_index+1] = 1 
      
      # Normalise
      vector = normalise(deepcopy(vector))
      
      # Update population
      pop[p]['vector'] = deepcopy(vector)      


    # --------------------
    # GUIDE E&E.
    if ee_guided is True:
      # Get the vectors.
      vectors = []
      for k in range(len(pop)):
        vectors.append(deepcopy(pop[k]['vector']))
      # Guide.
      vectors = guide_explo(deepcopy(vectors))
      # Update population vectors and scores.
      for k in range(len(vectors)):
        pop[k]['vector'] = deepcopy(vectors[k])
    # --------------------
    """
    # --------------------
    # EXPLO VARIATIONS WITHOUT GUIDING
    # Get the vectors.
    vectors = []
    for k in range(len(pop)):
      vectors.append(deepcopy(pop[k]['vector']))
    # Explo Variations.
    vectors = explo_var(deepcopy(vectors))
    # Update population vectors and scores.
    for k in range(len(vectors)):
      pop[k]['vector'] = deepcopy(vectors[k])
    # --------------------
    """
    

    
    # LAST TREATMENTS
    # Create the log structure for the iteration.
    ee_logs.append({
                  'iteration':i+1,
                  'vectors':[],
                  'vectors_quantity':0,
                  'top_quantity':0,
                  'top_vectors_score':0,
                  'all_vectors_coverage':0,
                  'top_vectors_constraints_completion':0,
                  'similarity_with_top_vectors':0,
                  'power':0,
                  'hvv':0,
                  'hvo':0,
                  'd%v':[],
                  'd%o':[],
                  'hvcv':0,
                  'hvco':0
                })
    for k in range(len(pop)):  
      # Update score
      vector_score = score(pop[k]['vector'])
      pop[k]['score'] = vector_score
      # Add to top_vectors and all_vectors.    
      add_to_top_vectors(deepcopy(pop[k]['vector']), vector_score)   
      all_vectors.append(deepcopy(pop[k]['vector']))

    # Feed E&E counters
    cause = 'Variations + Normalisation'
    iter = i+1
    feed_ee_counters_logged(deepcopy(pop), cause, iter)

    # CREATE POPULATION FOR NEXT ITERATION
    # Merge pop and pop_memory.
    pop += pop_memory
    # Sort population by score DESC.
    pop = sorted(deepcopy(pop), key=lambda dct: dct['score'], reverse=True)
    # Keep the best vectors by removing half of pop length.
    for k in range(population_size):
      pop.pop() # Remove last element from population (weakest one)
    

    # EVALUATIONS - ITERATION END 
    iteration_score = pop[0]['score']
    iteration_score_evolution.append(iteration_score)
    
    if iteration_score > max_score:
      max_score = iteration_score
      strongest_vector = deepcopy(pop[0]['vector'])
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:
    
      # --------------------
      # GUIDE E&E.
      if ee_guided is True:
        # Get the vectors.
        vectors = []
        for k in range(len(pop)):
          vectors.append(pop[k]['vector'])
        # Guide.
        vectors = guide_explo(vectors)
        # Update population vectors and scores.
        for k in range(len(vectors)):
          pop[k]['vector'] = vectors[k]
          pop[k]['score'] = score(vectors[k])
      # --------------------     
    
      # Convergence Management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score

    max_score_evolution.append(max_score)

  if convergence_score == 0: # if convergence not reached, convergence is considered to be the max_score.
      convergence_iteration = max_score_iteration
      convergence_score = max_score
      
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score, 
            res
          ]
          



# Extossom
# Extossom = Extract + Blossom
# General Principle : 
# Start with a population of vectors containing 1-digits randomly determined.
# ITERATION:
  # For each vector do :
    # EXTRACT: Select two random ranges in vector and keep the smallest one.
    # BLOSSOM on v1 : One on two digits in the range become a 1-digit. The FIRST digit is the first to be changed into a 1-digit. Create v1 as a copy of vector. It will receive the changes.
    # BLOSSOM on v2 : One on two digits in the range become a 1-digit. The SECOND digit is the first to be changed into a 1-digit. Create v2 as a copy of vector. It will receive the changes.
    # Keep the best BLOSSOM: Compare v1 and v2 score. vector become the best one.
    # Each 0-digit outside the range has a chance to become a 1-digit.
    # Normalise the vector.
    # Add to top_vectors and all_vectors.
  # Replace best vector if needed.
  # Create new population with the best vectors from past and new population for next iteration.
# The will behind this process is to do sorts of “divide into squares” variations on random ranges. We divide the selected range into two shapes having 1-digits one on two digits, and one shape being the reverse of the other. Doing so allows to not saturate the range and avoid lots of 1-digits removals by the normalisation function. By keeping the best shape among the two, we guarantee the best exploration possible of the selected range regarding this “divide into squares” feature. Moreover, doing additional variations outside the range allows to guarantee some more exploration. To sum up, this algorithm is mainly exploratory, nevertheless it performs some choices regarding this exploration and enrich the search space with light variations. We assume that this combination brings and sustain exploitation too.
def extossom():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  
  pop = []
  population_size = 2
  
  strongest_vector = []
  max_score = 0
  iteration_score_evolution = []
  max_score_evolution = []
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0


  # INITIALISATION
  # The initial population of vectors containing 1-digits randomly determined.
  for k in range(population_size):
    vector = generate_vector()
    vector_score = score(vector)
    pop.append({
                  'vector':vector,
                  'score':vector_score
              })
    # Add to top_vectors and all_vectors.   
    add_to_top_vectors(deepcopy(vector), vector_score)   
    all_vectors.append(deepcopy(vector))
    # Transfer newly discovered digits indexes from "ndd" to "dd".
    transfer_from_ndd_to_dd(deepcopy(vector))
    
 
  # ITERATIONS
  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations

    print("Iteration N° " + str(i+1) + " ", end="\r")
    
    # MEMORY
    pop_memory = deepcopy(pop)

    # For each vector in the population do :
    for p in range(len(pop)):
      vector = pop[p]['vector']
    
      # EXTRACT
      # Select two random ranges in vector and keep the smallest one.
      range_start_1 = random.randint(0, len(df)-2)
      range_end_1 = random.randint(range_start_1+1, len(df)-1)
      range_length_1 = range_end_1 - range_start_1
      range_start_2 = random.randint(0, len(df)-2)
      range_end_2 = random.randint(range_start_2+1, len(df)-1)
      range_length_2 = range_end_2 - range_start_2
      range_start = 0
      range_end = 0
      if range_length_1 <= range_length_2:
        range_start = range_start_1
        range_end = range_end_1
      else:
        range_start = range_start_2
        range_end = range_end_2
      
      # BLOSSOM on v1 : One on two digits in the range become a 1-digit.
      # The FIRST digit is the first to be changed into a 1-digit.
      # Create v1 as a copy of vector. It will receive the changes.
      v1 = deepcopy(vector)
      become_1_digit = True
      for k in range(range_start, range_end+1):
        if become_1_digit is True:
          v1[k] = 1 
          become_1_digit = False
        else:
          become_1_digit = True

      # BLOSSOM on v2 : One on two digits in the range become a 1-digit.
      # The SECOND digit is the first to be changed into a 1-digit.
      # Create v2 as a copy of vector. It will receive the changes.
      v2 = deepcopy(vector)
      become_1_digit = False
      for k in range(range_start, range_end+1):
        if become_1_digit is True:
          v2[k] = 1 
          become_1_digit = False
        else:
          become_1_digit = True
      
      # Keep the best BLOSSOM.
      # Compare v1 and v2 score. vector become the best one.
      if score(v1) >= score(v2):
        vector = deepcopy(v1)
      else:
        vector = deepcopy(v2)
      
      # Each 0-digit outside the range has a chance to become a 1-digit.
      chance_to_mute = 2 # Must be between 1 and 100.
      for k in range(0, range_start):
        if vector[k] == 0:
          rand = random.randint(1, 100)
          if rand < chance_to_mute: 
            vector[k] = 1
      for k in range(range_end+1, len(vector)):
        if vector[k] == 0:
          rand = random.randint(1, 100)
          if rand < chance_to_mute:
            vector[k] = 1


      # Normalise
      vector = normalise(deepcopy(vector))
      
      # Update population
      pop[p]['vector'] = deepcopy(vector)      

    # --------------------
    # GUIDE E&E.
    if ee_guided is True:
      # Get the vectors.
      vectors = []
      for k in range(len(pop)):
        vectors.append(deepcopy(pop[k]['vector']))
      # Guide.
      vectors = guide_explo(deepcopy(vectors))
      # Update population vectors and scores.
      for k in range(len(vectors)):
        pop[k]['vector'] = deepcopy(vectors[k])
    # --------------------
    
    """
    # --------------------
    # EXPLO VARIATIONS WITHOUT GUIDING
    # Get the vectors.
    vectors = []
    for k in range(len(pop)):
      vectors.append(deepcopy(pop[k]['vector']))
    # Explo Variations.
    vectors = explo_var(deepcopy(vectors))
    # Update population vectors and scores.
    for k in range(len(vectors)):
      pop[k]['vector'] = deepcopy(vectors[k])
    # --------------------
    """
    
    # Feed E&E counters
    feed_ee_counters(deepcopy(pop))
    
    # LAST TREATMENTS
    for k in range(len(pop)):  
      # Update score
      vector_score = score(pop[k]['vector'])
      pop[k]['score'] = vector_score
      # Add to top_vectors and all_vectors.    
      add_to_top_vectors(deepcopy(pop[k]['vector']), vector_score)
      all_vectors.append(deepcopy(pop[k]['vector']))


    # CREATE POPULATION FOR NEXT ITERATION
    # Merge pop and pop_memory.
    pop += pop_memory
    # Sort population by score DESC.
    pop = sorted(deepcopy(pop), key=lambda dct: dct['score'], reverse=True)
    # Keep the best vectors by removing half of pop length.
    for k in range(population_size):
      pop.pop() # Remove last element from population (weakest one)


    # EVALUATIONS - ITERATION END 
    iteration_score = pop[0]['score']
    iteration_score_evolution.append(iteration_score)
    
    if iteration_score > max_score:
      max_score = iteration_score
      strongest_vector = deepcopy(pop[0]['vector'])
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:
      # Convergence Management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score

    max_score_evolution.append(max_score)

  if convergence_score == 0: # if convergence not reached, convergence is considered to be the max_score.
      convergence_iteration = max_score_iteration
      convergence_score = max_score
      
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score, 
            res
          ]




def extossom_ee_logged():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  
  pop = []
  population_size = 2
  
  strongest_vector = []
  max_score = 0
  iteration_score_evolution = []
  max_score_evolution = []
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0


  # INITIALISATION
  # Create the log structure for the iteration (here it is iteration number 0 because we are initializing the population).
  ee_logs.append({
                'iteration':0,
                'vectors':[],
                'vectors_quantity':0,
                'top_quantity':0,
                'top_vectors_score':0,
                'all_vectors_coverage':0,
                'top_vectors_constraints_completion':0,
                'similarity_with_top_vectors':0,
                'power':0,
                'hvv':0,
                'hvo':0,
                'd%v':[],
                'd%o':[],
                'hvcv':0,
                'hvco':0
              })
  # The initial population of vectors containing 1-digits randomly determined.
  for k in range(population_size):
    vector = generate_vector()
    vector_score = score(vector)
    pop.append({
                  'vector':vector,
                  'score':vector_score
              })
    # Add to top_vectors and all_vectors.   
    add_to_top_vectors(deepcopy(vector), vector_score)   
    all_vectors.append(deepcopy(vector))

  # Feed ee_counters and the other attributes of the iteration log.
  feed_ee_counters_logged(deepcopy(pop), 'Pure Random Function', 0) 
    
 
  # ITERATIONS
  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations

    print("Iteration N° " + str(i+1) + " ", end="\r")
    
    # MEMORY
    pop_memory = deepcopy(pop)

    # For each vector in the population do :
    for p in range(len(pop)):
      vector = pop[p]['vector']
    
      # EXTRACT
      # Select two random ranges in vector and keep the smallest one.
      range_start_1 = random.randint(0, len(df)-2)
      range_end_1 = random.randint(range_start_1+1, len(df)-1)
      range_length_1 = range_end_1 - range_start_1
      range_start_2 = random.randint(0, len(df)-2)
      range_end_2 = random.randint(range_start_2+1, len(df)-1)
      range_length_2 = range_end_2 - range_start_2
      range_start = 0
      range_end = 0
      if range_length_1 <= range_length_2:
        range_start = range_start_1
        range_end = range_end_1
      else:
        range_start = range_start_2
        range_end = range_end_2
      
      # BLOSSOM on v1 : One on two digits in the range become a 1-digit.
      # The FIRST digit is the first to be changed into a 1-digit.
      # Create v1 as a copy of vector. It will receive the changes.
      v1 = deepcopy(vector)
      become_1_digit = True
      for k in range(range_start, range_end+1):
        if become_1_digit is True:
          v1[k] = 1 
          become_1_digit = False
        else:
          become_1_digit = True

      # BLOSSOM on v2 : One on two digits in the range become a 1-digit.
      # The SECOND digit is the first to be changed into a 1-digit.
      # Create v2 as a copy of vector. It will receive the changes.
      v2 = deepcopy(vector)
      become_1_digit = False
      for k in range(range_start, range_end+1):
        if become_1_digit is True:
          v2[k] = 1 
          become_1_digit = False
        else:
          become_1_digit = True
      
      # Keep the best BLOSSOM.
      # Compare v1 and v2 score. vector become the best one.
      if score(v1) >= score(v2):
        vector = deepcopy(v1)
      else:
        vector = deepcopy(v2)
      
      # Each 0-digit outside the range has a chance to become a 1-digit.
      chance_to_mute = 2 # Must be between 1 and 100.
      for k in range(0, range_start):
        if vector[k] == 0:
          rand = random.randint(1, 100)
          if rand < chance_to_mute: 
            vector[k] = 1
      for k in range(range_end+1, len(vector)):
        if vector[k] == 0:
          rand = random.randint(1, 100)
          if rand < chance_to_mute:
            vector[k] = 1


      # Normalise
      vector = normalise(deepcopy(vector))
      
      # Update population
      pop[p]['vector'] = deepcopy(vector)      

    # --------------------
    # GUIDE E&E.
    if ee_guided is True:
      # Get the vectors.
      vectors = []
      for k in range(len(pop)):
        vectors.append(deepcopy(pop[k]['vector']))
      # Guide.
      vectors = guide_explo(deepcopy(vectors))
      # Update population vectors and scores.
      for k in range(len(vectors)):
        pop[k]['vector'] = deepcopy(vectors[k])
    # --------------------
    
    """
    # --------------------
    # EXPLO VARIATIONS WITHOUT GUIDING
    # Get the vectors.
    vectors = []
    for k in range(len(pop)):
      vectors.append(deepcopy(pop[k]['vector']))
    # Explo Variations.
    vectors = explo_var(deepcopy(vectors))
    # Update population vectors and scores.
    for k in range(len(vectors)):
      pop[k]['vector'] = deepcopy(vectors[k])
    # --------------------
    """
    

    
    # LAST TREATMENTS
    # Create the log structure for the iteration.
    ee_logs.append({
                  'iteration':i+1,
                  'vectors':[],
                  'vectors_quantity':0,
                  'top_quantity':0,
                  'top_vectors_score':0,
                  'all_vectors_coverage':0,
                  'top_vectors_constraints_completion':0,
                  'similarity_with_top_vectors':0,
                  'power':0,
                  'hvv':0,
                  'hvo':0,
                  'd%v':[],
                  'd%o':[],
                  'hvcv':0,
                  'hvco':0
                })
    for k in range(len(pop)):
      # Update score
      vector_score = score(pop[k]['vector'])
      pop[k]['score'] = vector_score
      # Add to top_vectors and all_vectors.    
      add_to_top_vectors(deepcopy(pop[k]['vector']), vector_score)
      all_vectors.append(deepcopy(pop[k]['vector']))

    # Feed E&E counters
    cause = 'Variations + Normalisation'
    iter = i+1
    feed_ee_counters_logged(deepcopy(pop), cause, iter)

    # CREATE POPULATION FOR NEXT ITERATION
    # Merge pop and pop_memory.
    pop += pop_memory
    # Sort population by score DESC.
    pop = sorted(deepcopy(pop), key=lambda dct: dct['score'], reverse=True)
    # Keep the best vectors by removing half of pop length.
    for k in range(population_size):
      pop.pop() # Remove last element from population (weakest one)


    # EVALUATIONS - ITERATION END 
    iteration_score = pop[0]['score']
    iteration_score_evolution.append(iteration_score)
    
    if iteration_score > max_score:
      max_score = iteration_score
      strongest_vector = deepcopy(pop[0]['vector'])
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:
      # Convergence Management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score

    max_score_evolution.append(max_score)

  if convergence_score == 0: # if convergence not reached, convergence is considered to be the max_score.
      convergence_iteration = max_score_iteration
      convergence_score = max_score
      
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score, 
            res
          ]




# U3S = Unexploited + Surrounding Shot + Symmetrical
# General Principle : 
# Start with a population of vectors containing 1-digits randomly assigned.
# ITERATION:
  # For each vector in the population do :
    # --------------------
    # Surrounding Shot
    # --------------------
    # Select a 1-digit.
    # For the selected 1-digit, look for the next 1-digit.
    # Digits around the next 1-digit become 1-digits (right and left).
    # --------------------
    # Symmetrical
    # Middle point symmetrical mutations.
    # --------------------
    # 1- Select the middle index in the vector.
    # 2- Select a random index in the vector, the digit at this index becomes a 1-digit and the digit symmetrically opposed to it regarding the middle index becomes a 1-digit too.
    # --------------------
    # Unexploited
    # --------------------
    # Change an unexploited digit (among population) into a 1-digit in each vector.
    # --------------------
    # Last Treatments
    # --------------------
    # Normalise the vector.
    # Update vector score.
    # Add to top_vectors and all_vectors.
  # Replace the best vector if needed.
  # Create new population with the best vectors from past and new population for next iteration.
def u3s():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  
  strongest_vector = []
  max_score = 0
  iteration_score_evolution = []
  max_score_evolution = []
  
  population = [] 
  population_size = 2
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0

  # INITIALIZATION
  # Generate the initial population.
  for i in range(0, population_size):
    vector = generate_vector()
    vector_score = score(deepcopy(vector))
    # Add the vector to the population.
    population.append({
                        'vector':deepcopy(vector),
                        'score':vector_score
                      }) 
    # Add to top_vectors and all_vectors.
    add_to_top_vectors(deepcopy(vector), vector_score)   
    all_vectors.append(deepcopy(vector))
    # Transfer newly discovered digits indexes from "ndd" to "dd".
    transfer_from_ndd_to_dd(deepcopy(vector))
    
 
  # ITERATIONS  
  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations
    
    print("Iteration N° " + str(i+1) + " ", end="\r")
    
    # Memory of population
    population_memory = deepcopy(population)

    # ----------
    # MUTATIONS   
    # ----------  

    # ------------------------------
    # Surrounding Shot
    # For each vector :
    #   Select a 1-digit.
    #   For the selected 1-digit, look for the next 1-digit.
    #   Digits around the next 1-digit become 1-digits (right and left).
    #   If max length is reached and next_index not found, continue from index 0 in the vector.
    #   If next_index=0, left_index become the max_index.
    #   If next_index=max_index, right_index become 0.
    for k in range(len(population)):
      original_index = 0 # index of the considered 1-digit.
      next_index = 0 # index of the target 1-digit.
      new_range_start = 0
      # Pick the index of a random 1-digit.
      one_digits_indexes = []
      for d in range(len(population[k]['vector'])):
        if population[k]['vector'][d] == 1:
          one_digits_indexes.append(d)
      original_index = random.choice(one_digits_indexes)
      new_range_start = original_index + 1
      if original_index == len(population[k]['vector'])-1:
        new_range_start = 0
      # Pick the index of the next 1-digit.
      found = False
      while found is False:
        for d in range(new_range_start, len(population[k]['vector'])): 
          if found is False:
            if population[k]['vector'][d] == 1:
              next_index = d
              found = True
        if found is False:
          new_range_start = 0
      # Right and left indexes.
      right_index = next_index+1
      left_index = next_index-1
      if next_index == 0:
        left_index = len(df)-1
      elif next_index == len(df)-1:
        right_index = 0
      # Digits around next_index become 1-digits (right and left).
      population[k]['vector'][right_index] = 1
      population[k]['vector'][left_index] = 1

    # ------------------------------
    # Symmetrical
    # Middle point symmetrical mutations on each vector of the population.
    # 1- Select the middle index in the vector.
    # 2- Select a random index in the vector, the digit at this index becomes a 1-digit and the digit symmetrically opposed to it regarding the middle index becomes a 1-digit too.
    middle_index = int((len(df)-1)/2)
    for k in range(len(population)):
      ri = random.randint(0, len(df)-1)
      population[k]['vector'][ri] = 1 
      if ri > middle_index:
        target_index = middle_index - ri
        if target_index >= 0:
          population[k]['vector'][target_index] = 1     
      else:
        target_index = middle_index + ri
        if target_index < len(df):
          population[k]['vector'][target_index] = 1  

    # ------------------------------
    # Unexploited
    # Change an unexploited digit (among population) into a 1-digit in each vector.
    # Get the 0-digits indexes among all vectors of population.
    vectors = []
    for k in range(len(population)):
      vectors.append(population[k]['vector'])
    summed_vectors = sum(vectors, axis=0)
    zero_digits_indexes = []
    for k in range(len(summed_vectors)):
      if summed_vectors[k] == 0:
        zero_digits_indexes.append(k)
    for k in range(len(population)):
      # A random 0-digit becomes a 1-digit if this digit hasn't already been chosen for another vector in zero_digits_indexes
      if len(zero_digits_indexes) > 0:
        rand = random.choice(zero_digits_indexes)
        population[k]['vector'][rand] = 1
        zero_digits_indexes.remove(rand)

    # ------------------------------
    # LAST TREATMENTS
    # Constraints check + scores update + appends   
    for k in range(len(population)):  
      # Assure that the vector respects constraints.
      population[k]['vector'] = normalise(deepcopy(population[k]['vector']))
      # Recalculate the score of the vector.
      population[k]['score'] = score(deepcopy(population[k]['vector']))
      # Add the vector in top_vectors and all_vectors.
      add_to_top_vectors(deepcopy(population[k]['vector']), population[k]['score'])   
      all_vectors.append(deepcopy(population[k]['vector'])) 
      
    # Feed E&E counters
    feed_ee_counters(deepcopy(population))
      
    # ------------------------------
    # CREATE POPULATION FOR NEXT ITERATION
    # Merge population and population_memory.
    population += population_memory
    # Sort population by score DESC.
    population = sorted(deepcopy(population), key=lambda dct: dct['score'], reverse=True)
    # Keep the best vectors by removing half of population length.
    for k in range(population_size):
      population.pop() # Remove last element from population (weakest one)

    # ------------------------------
    # EVALUATIONS - ITERATION END       
    over_max_score = False
    power_k = population[0]['score'] 
    if power_k > max_score:
      over_max_score = True
      max_score = power_k
      strongest_vector = deepcopy(population[0]['vector'])
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
          
    if over_max_score is False:
      # Convergence Management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score
    
    max_score_evolution.append(max_score)
    iteration_score_evolution.append(power_k)

  if convergence_score == 0: # if convergence not reached, convergence is considered to be max_score.
    convergence_iteration = max_score_iteration
    convergence_score = max_score
    
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score,
            res
          ]




def u3s_ee_logged():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  
  strongest_vector = []
  max_score = 0
  iteration_score_evolution = []
  max_score_evolution = []
  
  population = [] 
  population_size = 2
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0

  # INITIALIZATION
  # Create the log structure for the iteration (here it is iteration number 0 because we are initializing the population).
  ee_logs.append({
                'iteration':0,
                'vectors':[],
                'vectors_quantity':0,
                'top_quantity':0,
                'top_vectors_score':0,
                'all_vectors_coverage':0,
                'top_vectors_constraints_completion':0,
                'similarity_with_top_vectors':0,
                'power':0,
                'hvv':0,
                'hvo':0,
                'd%v':[],
                'd%o':[],
                'hvcv':0,
                'hvco':0
              })
  # Generate the initial population.
  for i in range(0, population_size):
    vector = generate_vector()
    vector_score = score(deepcopy(vector))
    # Add the vector to the population.
    population.append({
                        'vector':deepcopy(vector),
                        'score':vector_score
                      }) 
    # Add to top_vectors and all_vectors.
    add_to_top_vectors(deepcopy(vector), vector_score)   
    all_vectors.append(deepcopy(vector))

  # Feed ee_counters and the other attributes of the iteration log.
  feed_ee_counters_logged(deepcopy(population), 'Pure Random Function', 0) 
    
 
  # ITERATIONS  
  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations
    
    print("Iteration N° " + str(i+1) + " ", end="\r")
    
    # Memory of population
    population_memory = deepcopy(population)

    # ----------
    # MUTATIONS   
    # ----------  

    # ------------------------------
    # Surrounding Shot
    # For each vector :
    #   Select a 1-digit.
    #   For the selected 1-digit, look for the next 1-digit.
    #   Digits around the next 1-digit become 1-digits (right and left).
    #   If max length is reached and next_index not found, continue from index 0 in the vector.
    #   If next_index=0, left_index become the max_index.
    #   If next_index=max_index, right_index become 0.
    for k in range(len(population)):
      original_index = 0 # index of the considered 1-digit.
      next_index = 0 # index of the target 1-digit.
      new_range_start = 0
      # Pick the index of a random 1-digit.
      one_digits_indexes = []
      for d in range(len(population[k]['vector'])):
        if population[k]['vector'][d] == 1:
          one_digits_indexes.append(d)
      original_index = random.choice(one_digits_indexes)
      new_range_start = original_index + 1
      if original_index == len(population[k]['vector'])-1:
        new_range_start = 0
      # Pick the index of the next 1-digit.
      found = False
      while found is False:
        for d in range(new_range_start, len(population[k]['vector'])): 
          if found is False:
            if population[k]['vector'][d] == 1:
              next_index = d
              found = True
        if found is False:
          new_range_start = 0
      # Right and left indexes.
      right_index = next_index+1
      left_index = next_index-1
      if next_index == 0:
        left_index = len(df)-1
      elif next_index == len(df)-1:
        right_index = 0
      # Digits around next_index become 1-digits (right and left).
      population[k]['vector'][right_index] = 1
      population[k]['vector'][left_index] = 1

    # ------------------------------
    # Symmetrical
    # Middle point symmetrical mutations on each vector of the population.
    # 1- Select the middle index in the vector.
    # 2- Select a random index in the vector, the digit at this index becomes a 1-digit and the digit symmetrically opposed to it regarding the middle index becomes a 1-digit too.
    middle_index = int((len(df)-1)/2)
    for k in range(len(population)):
      ri = random.randint(0, len(df)-1)
      population[k]['vector'][ri] = 1 
      if ri > middle_index:
        target_index = middle_index - ri
        if target_index >= 0:
          population[k]['vector'][target_index] = 1     
      else:
        target_index = middle_index + ri
        if target_index < len(df):
          population[k]['vector'][target_index] = 1  

    # ------------------------------
    # Unexploited
    # Change an unexploited digit (among population) into a 1-digit in each vector.
    # Get the 0-digits indexes among all vectors of population.
    vectors = []
    for k in range(len(population)):
      vectors.append(population[k]['vector'])
    summed_vectors = sum(vectors, axis=0)
    zero_digits_indexes = []
    for k in range(len(summed_vectors)):
      if summed_vectors[k] == 0:
        zero_digits_indexes.append(k)
    for k in range(len(population)):
      # A random 0-digit becomes a 1-digit if this digit hasn't already been chosen for another vector in zero_digits_indexes
      if len(zero_digits_indexes) > 0:
        rand = random.choice(zero_digits_indexes)
        population[k]['vector'][rand] = 1
        zero_digits_indexes.remove(rand)

    # ------------------------------
    # LAST TREATMENTS
    # Create the log structure for the iteration.
    ee_logs.append({
                  'iteration':i+1,
                  'vectors':[],
                  'vectors_quantity':0,
                  'top_quantity':0,
                  'top_vectors_score':0,
                  'all_vectors_coverage':0,
                  'top_vectors_constraints_completion':0,
                  'similarity_with_top_vectors':0,
                  'power':0,
                  'hvv':0,
                  'hvo':0,
                  'd%v':[],
                  'd%o':[],
                  'hvcv':0,
                  'hvco':0
                })
    # Constraints check + scores update + appends   
    for k in range(len(population)):  
      # Assure that the vector respects constraints.
      population[k]['vector'] = normalise(deepcopy(population[k]['vector']))
      # Recalculate the score of the vector.
      population[k]['score'] = score(deepcopy(population[k]['vector']))
      # Add the vector in top_vectors and all_vectors.
      add_to_top_vectors(deepcopy(population[k]['vector']), population[k]['score'])   
      all_vectors.append(deepcopy(population[k]['vector'])) 
      
    # Feed E&E counters
    cause = 'Variations + Normalisation'
    iter = i+1
    feed_ee_counters_logged(deepcopy(population), cause, iter)
      
    # ------------------------------
    # CREATE POPULATION FOR NEXT ITERATION
    # Merge population and population_memory.
    population += population_memory
    # Sort population by score DESC.
    population = sorted(deepcopy(population), key=lambda dct: dct['score'], reverse=True)
    # Keep the best vectors by removing half of population length.
    for k in range(population_size):
      population.pop() # Remove last element from population (weakest one)

    # ------------------------------
    # EVALUATIONS - ITERATION END       
    over_max_score = False
    power_k = population[0]['score'] 
    if power_k > max_score:
      over_max_score = True
      max_score = power_k
      strongest_vector = deepcopy(population[0]['vector'])
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
          
    if over_max_score is False:
      # Convergence Management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score
    
    max_score_evolution.append(max_score)
    iteration_score_evolution.append(power_k)

  if convergence_score == 0: # if convergence not reached, convergence is considered to be max_score.
    convergence_iteration = max_score_iteration
    convergence_score = max_score
    
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score,
            res
          ]




# EEP = E&E Power
# General Principle : 
# Start with a population of vectors containing 1-digits randomly assigned.
# ITERATION:
  # Calculate the indicator which is the mean of three percentages calculated with the six E&E power global variables.
  # Set the array to save indicator values (used to adjust the limits).
  # Set the limits of the indicator and set the tolerance spread around the mean indicator.
  # While indicator <= limit min
    # Force exploration by applying a variation operator on each vector of the population.
    # Adjust the indicator limits.
    # Update the indicator and save it.
  # While indicator >= limit max
    # Force exploitation by applying a variation operator on each vector of the population.
    # Adjust the indicator limits.
    # Update the indicator and save it.
  # LAST TREATMENTS - For each vector in the population do:
    # Normalise the vector.
    # Update vector score.
    # Add the vector in top_vectors and all_vectors.
  # Feed E&E counters.
  # Create population for next iteration.
  # Replace the best vector if needed.
def eep():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  
  global counter_exploit_iter
  global counter_exploit_vector
  global counter_exploit_digit
  
  strongest_vector = []
  max_score = 0
  iteration_score_evolution = []
  max_score_evolution = []
  
  population = [] 
  population_size = 2
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0

  # INITIALIZATION
  # Generate the initial population.
  for i in range(0, population_size):
    vector = generate_vector()
    vector_score = score(deepcopy(vector))
    # Add the vector to the population.
    population.append({
                        'vector':deepcopy(vector),
                        'score':vector_score
                      }) 
    # Add to top_vectors and all_vectors.
    add_to_top_vectors(deepcopy(vector), vector_score)   
    all_vectors.append(deepcopy(vector))
    # Transfer newly discovered digits indexes from "ndd" to "dd".
    transfer_from_ndd_to_dd(deepcopy(vector))
    
  # Initialise some counters to avoid division by zero error.
  counter_exploit_iter = 1
  counter_exploit_vector = 1
  counter_exploit_digit = 1
 
  # ITERATIONS  
  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations
    
    print("Iteration N° " + str(i+1) + " ", end="\r")
    
    # Memory of population
    population_memory = deepcopy(population)

    # ----------
    # VARIATIONS  
    # ----------
    
    # Calculate the indicator.
    # The indicator will be the mean of three percentages calculated with the six E&E power global variables.
    # NOTA : percentage_3 is multiplied by 1000 to make its value in the same height order as the two other percentages.
    percentage_1 = counter_explore_iter / counter_exploit_iter
    percentage_2 = counter_explore_vector / counter_exploit_vector
    percentage_3 = (counter_explore_digit / counter_exploit_digit)*1000
    ee_power_indicator = mean([percentage_1, percentage_2, percentage_3])
    
    # Save indicator values in order to adjust e_e_min and e_e_max.
    ee_power_indicator_values = []
    ee_power_indicator_values.append(ee_power_indicator)
    
    # Starting limits min and max for the indicator + tolerance spread around the mean indicator.
    e_e_min = 0.05
    e_e_max = 0.05
    spread = 0.01
    
    # Counters of values to add to E&E power variables when we update the percentages to update the indicator (see inside while loops below).
    # Everytime we force an exploration or exploitation, we will increment some of those counters.
    add_to_counter_explore_iter = 0
    add_to_counter_exploit_iter = 0
    add_to_counter_explore_vector = 0
    add_to_counter_exploit_vector = 0
    add_to_counter_explore_digit = 0
    add_to_counter_exploit_digit = 0


    while ee_power_indicator <= e_e_min:

      #print("===> FORCE EXPLORATION - Indicator= " + str(round(ee_power_indicator, 2)) + " ", end="\r")
      #print("===> FORCE EXPLORATION - Indicator= " + str(round(ee_power_indicator, 2)))
      
      add_to_counter_explore_iter += 1
      
      # Do variations to force exploration.
      # For each vector in the population, we randomly select a digit index from ndd (non-discovered digits) and we put it to 1 in the vector. If ndd is empty, we randomly select a digit index from dd (already discovered digits).
      for x in range(5):  
        for k in range(len(population)):
          rand_index = 0
          if len(ndd) > 0:
            rand_index = random.choice(ndd)
          else:
            rand_index = random.choice(dd)
          population[k]['vector'][rand_index] = 1
          add_to_counter_explore_vector += 1
          add_to_counter_explore_digit += 1
        
      # Adjust e_e_min and e_e_max using the tolerance spread toward mean ee_power_indicator.
      mean_ee_power_indicator = mean(ee_power_indicator_values)
      e_e_min = mean_ee_power_indicator - spread
      e_e_max = mean_ee_power_indicator + spread
      
      # Update ee_power_indicator and add it to ee_power_indicator_values.
      percentage_1 = (counter_explore_iter + add_to_counter_explore_iter) / (counter_exploit_iter + add_to_counter_exploit_iter)
      percentage_2 = (counter_explore_vector + add_to_counter_explore_vector) / (counter_exploit_vector + add_to_counter_exploit_vector)
      percentage_3 = ((counter_explore_digit + add_to_counter_explore_digit) / (counter_exploit_digit + add_to_counter_exploit_digit))*1000
      ee_power_indicator = mean([percentage_1, percentage_2, percentage_3])
      ee_power_indicator_values.append(ee_power_indicator)


    while ee_power_indicator >= e_e_max:

      #print("===> FORCE EXPLOITATION - Indicator= " + str(round(ee_power_indicator, 2)) + " ", end="\r")
      #print("===> FORCE EXPLOITATION - Indicator= " + str(round(ee_power_indicator, 2)))
      
      add_to_counter_exploit_iter += 1
      
      # Do variations to force exploitation.
      # For each vector in the population, we randomly select a digit index from dd (already discovered digits) and we put it to 1 in the vector.
      for x in range(5):
        for k in range(len(population)):
          rand_index = random.choice(dd)
          population[k]['vector'][rand_index] = 1
          add_to_counter_exploit_vector += 1
          add_to_counter_exploit_digit += 1
        
      # Adjust e_e_min and e_e_max using the tolerance spread toward mean ee_power_indicator.
      mean_ee_power_indicator = mean(ee_power_indicator_values)
      e_e_min = mean_ee_power_indicator - spread
      e_e_max = mean_ee_power_indicator + spread
      
      # Update ee_power_indicator and add it to ee_power_indicator_values.
      percentage_1 = (counter_explore_iter + add_to_counter_explore_iter) / (counter_exploit_iter + add_to_counter_exploit_iter)
      percentage_2 = (counter_explore_vector + add_to_counter_explore_vector) / (counter_exploit_vector + add_to_counter_exploit_vector)
      percentage_3 = ((counter_explore_digit + add_to_counter_explore_digit) / (counter_exploit_digit + add_to_counter_exploit_digit))*1000
      ee_power_indicator = mean([percentage_1, percentage_2, percentage_3])
      ee_power_indicator_values.append(ee_power_indicator)


    # ------------------------------
    # LAST TREATMENTS
    # Constraints check + scores update + appends   
    for k in range(len(population)):  
      # Assure that the vector respects constraints.
      population[k]['vector'] = normalise(deepcopy(population[k]['vector']))
      # Recalculate the score of the vector.
      population[k]['score'] = score(deepcopy(population[k]['vector']))
      # Add the vector in top_vectors and all_vectors.
      add_to_top_vectors(deepcopy(population[k]['vector']), population[k]['score'])   
      all_vectors.append(deepcopy(population[k]['vector']))
    
    # Feed E&E counters
    feed_ee_counters(deepcopy(population))
      
    # ------------------------------
    # CREATE POPULATION FOR NEXT ITERATION
    # Merge population and population_memory.
    population += population_memory
    # Sort population by score DESC.
    population = sorted(deepcopy(population), key=lambda dct: dct['score'], reverse=True)
    # Keep the best vectors by removing half of population length.
    for k in range(population_size):
      population.pop() # Remove last element from population (weakest one)

    # ------------------------------
    # EVALUATIONS - ITERATION END       
    over_max_score = False
    power_k = population[0]['score'] 
    if power_k > max_score:
      over_max_score = True
      max_score = power_k
      strongest_vector = deepcopy(population[0]['vector'])
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
          
    if over_max_score is False:
      # Convergence Management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score
    
    max_score_evolution.append(max_score)
    iteration_score_evolution.append(power_k)

  if convergence_score == 0: # if convergence not reached, convergence is considered to be max_score.
    convergence_iteration = max_score_iteration
    convergence_score = max_score
    
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score,
            res
          ]




def eep_ee_logged():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  
  global counter_exploit_iter
  global counter_exploit_vector
  global counter_exploit_digit
  
  strongest_vector = []
  max_score = 0
  iteration_score_evolution = []
  max_score_evolution = []
  
  population = [] 
  population_size = 2
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0

  # INITIALIZATION
  # Create the log structure for the iteration (here it is iteration number 0 because we are initializing the population).
  ee_logs.append({
                'iteration':0,
                'vectors':[],
                'vectors_quantity':0,
                'top_quantity':0,
                'top_vectors_score':0,
                'all_vectors_coverage':0,
                'top_vectors_constraints_completion':0,
                'similarity_with_top_vectors':0,
                'power':0,
                'hvv':0,
                'hvo':0,
                'd%v':[],
                'd%o':[],
                'hvcv':0,
                'hvco':0
              })
  # Generate the initial population.
  for i in range(0, population_size):
    vector = generate_vector()
    vector_score = score(deepcopy(vector))
    # Add the vector to the population.
    population.append({
                        'vector':deepcopy(vector),
                        'score':vector_score
                      }) 
    # Add to top_vectors and all_vectors.
    add_to_top_vectors(deepcopy(vector), vector_score)   
    all_vectors.append(deepcopy(vector))

  # Feed ee_counters and the other attributes of the iteration log.
  feed_ee_counters_logged(deepcopy(population), 'Pure Random Function', 0)
    
  # Initialise some counters to avoid division by zero error.
  counter_exploit_iter = 1
  counter_exploit_vector = 1
  counter_exploit_digit = 1
 
  # ITERATIONS  
  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations
    
    print("Iteration N° " + str(i+1) + " ", end="\r")
    
    # Memory of population
    population_memory = deepcopy(population)

    # ----------
    # VARIATIONS  
    # ----------
    
    # Calculate the indicator.
    # The indicator will be the mean of three percentages calculated with the six E&E power global variables.
    # NOTA : percentage_3 is multiplied by 1000 to make its value in the same height order as the two other percentages.
    percentage_1 = counter_explore_iter / counter_exploit_iter
    percentage_2 = counter_explore_vector / counter_exploit_vector
    percentage_3 = (counter_explore_digit / counter_exploit_digit)*1000
    ee_power_indicator = mean([percentage_1, percentage_2, percentage_3])
    
    # Save indicator values in order to adjust e_e_min and e_e_max.
    ee_power_indicator_values = []
    ee_power_indicator_values.append(ee_power_indicator)
    
    # Starting limits min and max for the indicator + tolerance spread around the mean indicator.
    e_e_min = 0.05
    e_e_max = 0.05
    spread = 0.01
    
    # Counters of values to add to E&E power variables when we update the percentages to update the indicator (see inside while loops below).
    # Everytime we force an exploration or exploitation, we will increment some of those counters.
    add_to_counter_explore_iter = 0
    add_to_counter_exploit_iter = 0
    add_to_counter_explore_vector = 0
    add_to_counter_exploit_vector = 0
    add_to_counter_explore_digit = 0
    add_to_counter_exploit_digit = 0


    while ee_power_indicator <= e_e_min:

      #print("===> FORCE EXPLORATION - Indicator= " + str(round(ee_power_indicator, 2)) + " ", end="\r")
      #print("===> FORCE EXPLORATION - Indicator= " + str(round(ee_power_indicator, 2)))
      
      add_to_counter_explore_iter += 1
      
      # Do variations to force exploration.
      # For each vector in the population, we randomly select a digit index from ndd (non-discovered digits) and we put it to 1 in the vector. If ndd is empty, we randomly select a digit index from dd (already discovered digits).
      for x in range(5):  
        for k in range(len(population)):
          rand_index = 0
          if len(ndd) > 0:
            rand_index = random.choice(ndd)
          else:
            rand_index = random.choice(dd)
          population[k]['vector'][rand_index] = 1
          add_to_counter_explore_vector += 1
          add_to_counter_explore_digit += 1
        
      # Adjust e_e_min and e_e_max using the tolerance spread toward mean ee_power_indicator.
      mean_ee_power_indicator = mean(ee_power_indicator_values)
      e_e_min = mean_ee_power_indicator - spread
      e_e_max = mean_ee_power_indicator + spread
      
      # Update ee_power_indicator and add it to ee_power_indicator_values.
      percentage_1 = (counter_explore_iter + add_to_counter_explore_iter) / (counter_exploit_iter + add_to_counter_exploit_iter)
      percentage_2 = (counter_explore_vector + add_to_counter_explore_vector) / (counter_exploit_vector + add_to_counter_exploit_vector)
      percentage_3 = ((counter_explore_digit + add_to_counter_explore_digit) / (counter_exploit_digit + add_to_counter_exploit_digit))*1000
      ee_power_indicator = mean([percentage_1, percentage_2, percentage_3])
      ee_power_indicator_values.append(ee_power_indicator)


    while ee_power_indicator >= e_e_max:

      #print("===> FORCE EXPLOITATION - Indicator= " + str(round(ee_power_indicator, 2)) + " ", end="\r")
      #print("===> FORCE EXPLOITATION - Indicator= " + str(round(ee_power_indicator, 2)))
      
      add_to_counter_exploit_iter += 1
      
      # Do variations to force exploitation.
      # For each vector in the population, we randomly select a digit index from dd (already discovered digits) and we put it to 1 in the vector.
      for x in range(5):
        for k in range(len(population)):
          rand_index = random.choice(dd)
          population[k]['vector'][rand_index] = 1
          add_to_counter_exploit_vector += 1
          add_to_counter_exploit_digit += 1
        
      # Adjust e_e_min and e_e_max using the tolerance spread toward mean ee_power_indicator.
      mean_ee_power_indicator = mean(ee_power_indicator_values)
      e_e_min = mean_ee_power_indicator - spread
      e_e_max = mean_ee_power_indicator + spread
      
      # Update ee_power_indicator and add it to ee_power_indicator_values.
      percentage_1 = (counter_explore_iter + add_to_counter_explore_iter) / (counter_exploit_iter + add_to_counter_exploit_iter)
      percentage_2 = (counter_explore_vector + add_to_counter_explore_vector) / (counter_exploit_vector + add_to_counter_exploit_vector)
      percentage_3 = ((counter_explore_digit + add_to_counter_explore_digit) / (counter_exploit_digit + add_to_counter_exploit_digit))*1000
      ee_power_indicator = mean([percentage_1, percentage_2, percentage_3])
      ee_power_indicator_values.append(ee_power_indicator)


    # ------------------------------
    # LAST TREATMENTS
    
    # Create the log structure for the iteration.
    ee_logs.append({
                  'iteration':i+1,
                  'vectors':[],
                  'vectors_quantity':0,
                  'top_quantity':0,
                  'top_vectors_score':0,
                  'all_vectors_coverage':0,
                  'top_vectors_constraints_completion':0,
                  'similarity_with_top_vectors':0,
                  'power':0,
                  'hvv':0,
                  'hvo':0,
                  'd%v':[],
                  'd%o':[],
                  'hvcv':0,
                  'hvco':0
                })
    
    # Constraints check + scores update + appends   
    for k in range(len(population)):  
      # Assure that the vector respects constraints.
      population[k]['vector'] = normalise(deepcopy(population[k]['vector']))
      # Recalculate the score of the vector.
      population[k]['score'] = score(deepcopy(population[k]['vector']))
      # Add the vector in top_vectors and all_vectors.
      add_to_top_vectors(deepcopy(population[k]['vector']), population[k]['score'])   
      all_vectors.append(deepcopy(population[k]['vector']))
    
    # Feed E&E counters
    cause = 'Variations + Normalisation'
    iter = i+1
    feed_ee_counters_logged(deepcopy(population), cause, iter)
      
    # ------------------------------
    # CREATE POPULATION FOR NEXT ITERATION
    # Merge population and population_memory.
    population += population_memory
    # Sort population by score DESC.
    population = sorted(deepcopy(population), key=lambda dct: dct['score'], reverse=True)
    # Keep the best vectors by removing half of population length.
    for k in range(population_size):
      population.pop() # Remove last element from population (weakest one)

    # ------------------------------
    # EVALUATIONS - ITERATION END       
    over_max_score = False
    power_k = population[0]['score'] 
    if power_k > max_score:
      over_max_score = True
      max_score = power_k
      strongest_vector = deepcopy(population[0]['vector'])
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
          
    if over_max_score is False:
      # Convergence Management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score
    
    max_score_evolution.append(max_score)
    iteration_score_evolution.append(power_k)

  if convergence_score == 0: # if convergence not reached, convergence is considered to be max_score.
    convergence_iteration = max_score_iteration
    convergence_score = max_score
    
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score,
            res
          ]
  



# Basic Reinforcement Learning Algorithm
def reinforcement():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  
  max_score = 0
  max_score_evolution = []
  iteration_score_evolution = []
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0
  
  # Number of digits in a vector
  d = len(df)
  
  # Reinforcement scores associated to each digit
  Q = [0 for _ in range(d)]

  # Number of times that each digit has been passed to 1
  N = [0 for _ in range(d)]

  # Exploration probability
  exprob = 0.2
  
  # INITIALIZATION
  """
  # Generate the initial vector randomly...
  vector = generate_vector()
  v_score = score(deepcopy(vector))
  if v_score > max_score:
    max_score = v_score
  """
  # ...or start with a vector full of 0-digits
  vector = []
  for k in range(len(df)):
    vector.append(0)
  v_score = 0
  
  """
  # Add in top_vectors and all_vectors.
  add_to_top_vectors(deepcopy(vector), v_score)   
  all_vectors.append(deepcopy(vector))
  # Transfer newly discovered digits indexes from "ndd" to "dd".
  transfer_from_ndd_to_dd(deepcopy(vector))
  """

  # ITERATIONS  
  iteration_number = algorithm_iterations
  for i in range(iteration_number):
    
    print("Iteration N° " + str(i+1) + " ", end="\r")

    vector_memory = deepcopy(vector)

    # Select x 0-digits indexes from vector. They will be changed into 1-digits.
    x = 2
    itc = [] # indexes to change
    for k in range(x):
      a = 0
      if random.uniform(0, 1) > exprob:
        # Take greedy action
        # Get the index of the best value found in Q.
        a = argmax(Q)
        # Assure that at selected index 'a' the value is 0.
        if vector[a] == 1 or a in itc:
          while vector[a] == 1 or a in itc:
            a = random.randint(0, d-1)
      else:
        # Take random action
        a = random.randint(0, d-1)
        # Assure that at selected index 'a' the value is 0.
        if vector[a] == 1 or a in itc:
          while vector[a] == 1 or a in itc:
            a = random.randint(0, d-1)
      itc.append(a)
    
    # Change digits at indexes from itc into 1-digits, normalise the vector.
    for k in range(x):
      vector[itc[k]] = 1
    vector = normalise(deepcopy(vector))


    # --------------------
    # GUIDE E&E.
    vectors = []
    if ee_guided is True:
      # Get the vectors.
      vectors.append(deepcopy(vector))
      
      # Guide.
      if guide_function == "separated_EE_cases":
        vectors = guide_explo_with_separated_EE_cases(deepcopy(vectors))
      elif guide_function == "merged_EE_cases":
        vectors = guide_explo_with_merged_EE_cases(deepcopy(vectors))
      elif guide_function == "potential":
        vectors = guide_explo_with_potential(deepcopy(vectors))
      elif guide_function == "cc":
        vectors = guide_explo_with_cc(deepcopy(vectors))
      elif guide_function == "links":
        vectors = guide_explo_with_links(deepcopy(vectors))
      elif guide_function == "vector_links":
        vectors = guide_explo_with_vector_links(deepcopy(vectors))
      elif guide_function == "power":
        vectors = guide_explo_with_power(deepcopy(vectors))
      elif guide_function == "temporal_balance":
        vectors = guide_explo_with_temporal_balance(deepcopy(vectors))
        
      # Update vector.
      vector = deepcopy(vectors[0])
    # --------------------      

    # LAST TREATMENTS
    # Feed E&E counters
    feed_ee_counters([{'vector':vector, 'score':None}])
    # Update score
    v_score = score(deepcopy(vector))
    # Add to top_vectors and all_vectors.
    add_to_top_vectors(deepcopy(vector), v_score)
    all_vectors.append(deepcopy(vector))
    # Add to vectors_classes.
    add_to_vectors_classes(deepcopy(vector), i)


    # EVALUATIONS - ITERATION END
    
    # Collect reward and update max_score
    over_max_score = False
    reward = 0 # Loose by default
    if v_score >= max_score:
      over_max_score = True
      reward = 1 # Win because a better score has been reached. 
      max_score = v_score
    else:
      vector = deepcopy(vector_memory)

    # Update 'N' and 'Q' for each index changed.
    for k in range(x):
      N[itc[k]] += 1
      Q[itc[k]] += 1/N[itc[k]] * (reward - Q[itc[k]])
    
    # Update max_score_evolution and iteration_score_evolution
    max_score_evolution.append(max_score)
    iteration_score_evolution.append(v_score)

    if over_max_score is True:
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:
      # Convergence Management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score

  if convergence_score == 0: # if convergence not yet reached, convergence is considered to be max_score.
    convergence_iteration = max_score_iteration
    convergence_score = max_score
    
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score, 
            res
          ]



def reinforcement_ee_logged():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  
  max_score = 0
  max_score_evolution = []
  iteration_score_evolution = []
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0
  
  # Number of digits in a vector
  d = len(df)
  
  # Reinforcement scores associated to each digit
  Q = [0 for _ in range(d)]

  # Number of times that each digit has been passed to 1
  N = [0 for _ in range(d)]

  # Exploration probability
  exprob = 0.2
  
  # INITIALIZATION
  """
  # Generate the initial vector randomly...
  vector = generate_vector()
  v_score = score(deepcopy(vector))
  if v_score > max_score:
    max_score = v_score
  """
  # ...or start with a vector full of 0-digits
  vector = []
  for k in range(len(df)):
    vector.append(0)
  v_score = 0
  
  """
  # Add in top_vectors and all_vectors.
  add_to_top_vectors(deepcopy(vector), v_score)   
  all_vectors.append(deepcopy(vector))
  # Transfer newly discovered digits indexes from "ndd" to "dd".
  transfer_from_ndd_to_dd(deepcopy(vector))
  """

  # ITERATIONS  
  iteration_number = algorithm_iterations
  for i in range(iteration_number):
    
    print("Iteration N° " + str(i+1) + " ", end="\r")

    vector_memory = deepcopy(vector)

    # Select x 0-digits indexes from vector. They will be changed into 1-digits.
    x = 2
    itc = [] # indexes to change
    for k in range(x):
      a = 0
      if random.uniform(0, 1) > exprob:
        # Take greedy action
        # Get the index of the best value found in Q.
        a = argmax(Q)
        # Assure that at selected index 'a' the value is 0.
        if vector[a] == 1 or a in itc:
          while vector[a] == 1 or a in itc:
            a = random.randint(0, d-1)
      else:
        # Take random action
        a = random.randint(0, d-1)
        # Assure that at selected index 'a' the value is 0.
        if vector[a] == 1 or a in itc:
          while vector[a] == 1 or a in itc:
            a = random.randint(0, d-1)
      itc.append(a)
    
    # Change digits at indexes from itc into 1-digits, normalise the vector.
    for k in range(x):
      vector[itc[k]] = 1
    vector = normalise(deepcopy(vector))


    # --------------------
    # GUIDE E&E.
    vectors = []
    if ee_guided is True:
      # Get the vectors.
      vectors.append(deepcopy(vector))
      
      # Guide.
      if guide_function == "separated_EE_cases":
        vectors = guide_explo_with_separated_EE_cases(deepcopy(vectors))
      elif guide_function == "merged_EE_cases":
        vectors = guide_explo_with_merged_EE_cases(deepcopy(vectors))
      elif guide_function == "potential":
        vectors = guide_explo_with_potential(deepcopy(vectors))
      elif guide_function == "cc":
        vectors = guide_explo_with_cc(deepcopy(vectors))
      elif guide_function == "links":
        vectors = guide_explo_with_links(deepcopy(vectors))
      elif guide_function == "vector_links":
        vectors = guide_explo_with_vector_links(deepcopy(vectors))
      elif guide_function == "power":
        vectors = guide_explo_with_power(deepcopy(vectors))
      elif guide_function == "temporal_balance":
        vectors = guide_explo_with_temporal_balance(deepcopy(vectors))
        
      # Update vector.
      vector = deepcopy(vectors[0])
    # --------------------      

    # LAST TREATMENTS
    # Create the log structure for the iteration.
    ee_logs.append({
                  'iteration':i,
                  'vectors':[],
                  'vectors_quantity':0,
                  'top_quantity':0,
                  'top_vectors_score':0,
                  'all_vectors_coverage':0,
                  'top_vectors_constraints_completion':0,
                  'similarity_with_top_vectors':0,
                  'power':0,
                  'hvv':0,
                  'hvo':0,
                  'd%v':[],
                  'd%o':[],
                  'hvcv':0,
                  'hvco':0
                })
    # Update score
    v_score = score(deepcopy(vector))
    # Add to top_vectors and all_vectors.
    add_to_top_vectors(deepcopy(vector), v_score)
    all_vectors.append(deepcopy(vector))
    # Add to vectors_classes.
    add_to_vectors_classes(deepcopy(vector), i)
    # Feed E&E counters
    cause = ''
    iter = i
    feed_ee_counters_logged([{'vector':vector, 'score':None}], cause, iter)


    # EVALUATIONS - ITERATION END
    
    # Collect reward and update max_score
    over_max_score = False
    reward = 0 # Loose by default
    if v_score >= max_score:
      over_max_score = True
      reward = 1 # Win because a better score has been reached. 
      max_score = v_score
    else:
      vector = deepcopy(vector_memory)

    # Update 'N' and 'Q' for each index changed.
    for k in range(x):
      N[itc[k]] += 1
      Q[itc[k]] += 1/N[itc[k]] * (reward - Q[itc[k]])
    
    # Update max_score_evolution and iteration_score_evolution
    max_score_evolution.append(max_score)
    iteration_score_evolution.append(v_score)

    if over_max_score is True:
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:
      # Convergence Management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score

  if convergence_score == 0: # if convergence not yet reached, convergence is considered to be max_score.
    convergence_iteration = max_score_iteration
    convergence_score = max_score
    
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score, 
            res
          ]




def combi_reduc():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  global df
  
  # Strongest vector
  v_strong = []
  # Score of the strongest vector
  max_score = 0
  
  iteration_score_evolution = []
  each_generation_mean_score = []
  max_score_evolution = []
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0


  
  # -------------------
  # REDUCTION SEQUENCES
  # -------------------
  # Final candidate vectors.
  candidate_vectors = []
  # Memorise full df.
  df_memory = deepcopy(df)
  # Declare the array of all reduced df obtained for further greedy treatments (see section "GREEDY TREATMENTS").
  all_reduced_df = []
  # Declare all reduction functions to be used.
  # "msm","dsm","ivs","mgtcm","shk","sk","csk"
  all_reduc_funcs = ["ivs","mgtcm","sk","csk"]
  
  # Create sequences of reduction functions, with "r" elements, "r" ranging from r_start to r_end. A function can be present (0, n) times in a sequence.
  # Sequence example : ["ivs","mgtcm","dsm"]
  r_start = 1
  r_end = 3
  sequences = []
  for r in range(r_start, r_end+1):
    sequens = list(product(all_reduc_funcs, repeat=r))
    for k in range(len(sequens)):
      sequences.append(sequens[k])
  
  # For each sequence, df is reduced following the order of reduction functions in the sequence.
  # The reduction is done while remaining resources in df do not allow to create a vector that respects constraints. The reduction come back to the first function of the sequence if needed.
  sequences_len = len(sequences)
  for t in range(sequences_len):
    
    print("Sequence N° " + str(t+1) + "/" + str(sequences_len), end="\r")

    sequ_index_max = len(sequences[t]) - 1
    sequ_index = 0

    # Initialise the vector corresponding to df size.
    vector = []
    for k in range(len(df)):
      vector.append(1)
    
    while constraints_max_values_check(vector) is False:
      
      # Execute the next reshape function.
      reshape_function = sequences[t][sequ_index]
      #ic(reshape_function)
      if reshape_function == "msm":
        reshape_mean_spp_mean()
      elif reshape_function == "dsm":
        reshape_detailed_spp_mean()
      elif reshape_function == "ivs":  
        reshape_intra_values_stability()
      elif reshape_function == "mgtcm":
        reshape_mean_gap_toward_columns_means()
      elif reshape_function == "sk":    
        reshape_score_kings()
      elif reshape_function == "shk":    
        reshape_score_hills_kings()
      elif reshape_function == "csk":    
        reshape_constraints_score_kings()
        
      # Adjust vector size to be the same as the new df.
      df_len = len(df)
      while len(vector) > df_len:
        vector.pop()
        
      # Increment or reinitialise sequ_index.
      if sequ_index == sequ_index_max:
        sequ_index = 0
      else:
        sequ_index += 1
    # while end
    
    # Save df obtained for further treatments.
    all_reduced_df.append(deepcopy(df))
    
    # Transform the vector in a conventional array of 0 and 1 digits so as to be able to normalise it.
    vector = []
    df_list = deepcopy(df).tolist()
    for k in range(len(df_memory)):
      if df_memory[k] in df_list:
        vector.append(1)
      else:
        vector.append(0)
    
    # Reinitialise df.
    df = deepcopy(df_memory)
    
    # Normalise the vector.
    vector = normalise(deepcopy(vector))
    
    # Calculate the vector score.
    vector_score = score(vector)
    
    # Save in candidate_vectors.
    candidate_vectors.append({
                                'vector':vector,
                                'score':vector_score,
                                'mother_reduc_sequ':sequences[t]
                            })
  
  # Add to top_vectors and all_vectors.
  for k in range(len(candidate_vectors)):
    add_to_top_vectors(deepcopy(candidate_vectors[k]['vector']), deepcopy(candidate_vectors[k]['score']))
    all_vectors.append(deepcopy(candidate_vectors[k]['vector']))
  
  # In fine, we have as many candidate vectors as reduction sequences.
  # The vector having the best score among candidate vectors is recommended.
  # Sort candidate_vectors by score DESC.
  candidate_vectors = sorted(deepcopy(candidate_vectors), key=lambda dct: dct['score'], reverse=True)
  # Get the best score.
  best_score = candidate_vectors[0]['score']
  # Get the associated best reduction sequence.
  brs = candidate_vectors[0]['mother_reduc_sequ']
  
  # Prints
  print("-----------------------------------")
  print("Best Reduc Sequence : " + str(brs))
  print("---- SCORES ----")
  print("Post Reduc : " + str(round(best_score,1)))
  
  
  # -------
  # MUTANTS
  # -------
  # Declare the array of all mutants.
  mutants = []
  
  # Perform mutations on the best x candidate vectors.
  x = 3
  for k in range(x):
    print("Vector N° " + str(k+1) + "/" + str(len(candidate_vectors)), end="\r")
    c_vector = candidate_vectors[k]['vector']
    # For each 0-digit in c_vector, create a copy of c_vector with this digit mutated in a 1-digit. Normalise, evaluate score, save.
    for d in range(len(c_vector)):
      if c_vector[d] == 0:
        mutant = deepcopy(c_vector)
        mutant[d] = 1
        mutant = normalise(deepcopy(mutant))
        mutants.append({
                          'vector':mutant,
                          'score':score(deepcopy(mutant))
                      })

  # Add to top_vectors and all_vectors.
  for k in range(len(mutants)):
    add_to_top_vectors(deepcopy(mutants[k]['vector']), deepcopy(mutants[k]['score']))
    all_vectors.append(deepcopy(mutants[k]['vector']))
    
  # The best mutant is recommended.
  # Sort mutants by score DESC.
  mutants = sorted(deepcopy(mutants), key=lambda dct: dct['score'], reverse=True)
  # Get the best score.
  best_score = mutants[0]['score']
  
  # Prints
  print("Post Muta : " + str(round(best_score,1)))
  print("-----------------------------------")
  

  # Feed E&E counters 
  # => Not taken into account, fed with three '0' for the compliance with the overall process.
  explore_iter_timers.append(0)
  explore_iter_timers.append(0)
  explore_iter_timers.append(0)

  # Strongest vector (v_strong) and associated score (max_score) 
  if candidate_vectors[0]['score'] >= mutants[0]['score']:
    max_score = candidate_vectors[0]['score']
    v_strong = candidate_vectors[0]['vector']
  else:
    max_score = mutants[0]['score']
    v_strong = mutants[0]['vector']
  
  # Convergence score
  convergence_score = max_score
    
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution, convergence_iteration)

  return [
            convergence_iteration, 
            convergence_score, 
            max_score_iteration, 
            max_score, 
            res
          ]






  
  """ DOES NOT WORK
  # ------------------------
  # FUSE - NORMALISE
  # ------------------------
  # Reconstruct a df (called rdf) by placing data occurrences from all_reduced_df only once in rdf.
  rdf = []
  for k in range(len(all_reduced_df)):
    for i in range(len(all_reduced_df[k])):
      if list(all_reduced_df[k][i]) not in rdf:
        rdf.append(list(all_reduced_df[k][i]))
  
  # Create the vector corresponding to rdf.
  vector = []
  for m in range(len(df_memory)):
    if df_memory[m] in rdf:
      vector.append(1)
    else:
      vector.append(0)
  
  # Normalise the vector n times.
  n = 1000
  norm_vectors = []
  for k in range(n):
    print("Normalise N° " + str(k+1) + "/" + str(n), end="\r")
    norm_vector = normalise(deepcopy(vector))
    # Evaluate vector score.
    norm_vector_score = score(norm_vector)
    # Save the vector.
    norm_vectors.append({
                            'vector':norm_vector,
                            'score':norm_vector_score
                        })
                          
  # The vector having the best score in norm_vectors is recommended.
  # Sort norm_vectors by score DESC.
  norm_vectors = sorted(deepcopy(norm_vectors), key=lambda dct: dct['score'], reverse=True)
  # Get the best score.
  best_score = norm_vectors[0]['score']
  
  # Prints
  print("-----------------------------------")
  print("----- AFTER FUSE & NORMALISE -----")
  print("Best Vector Score = " + str(round(best_score,1)))
  print("-----------------------------------")
  
  sys.exit()
  """
  
  
  
  """ 
  # DOES NOT WORK - It is normal because removing an element by considering combinations with size len(df)-1 has a huge impact on the score. Indeed this "removed" element is replaced by other ones during normalisation and these other ones have a great chance to not increase the score.
  # -----------------
  # GREEDY TREATMENTS
  # -----------------
  # This section aims to create, for each df in all_reduced_df, all combinations possible with size len(df)-1. In a df from all_reduced_df, all resources can be selected to create a vector that respects constraints (see above treatments). So we can produce combinations of size len(df)-1 in order to create vectors from the df, normalise them and evaluate their scores.
  final_vectors = []
  # For each df in all_reduced_df.
  for k in range(len(all_reduced_df)):
    # Get the df we will use in that iteration.
    treated_df = all_reduced_df[k]
    # Create all combinations of items from treated_df, with size len(treated_df)-1.
    r = len(treated_df) - 1
    combis = list(combinations(treated_df, r))
    # For each combi create the associated vector with 0 and 1 digits.
    for combi in range(len(combis)): 
      vector = []
      combi_list = list(deepcopy(combis[combi]))
      for cl in range(len(combi_list)):
        combi_list[cl] = list(combi_list[cl])
      for m in range(len(df_memory)):
        if df_memory[m] in combi_list:
          vector.append(1)
        else:
          vector.append(0)
      # Normalise the vector.
      vector = normalise(deepcopy(vector))
      # Evaluate vector score.
      vector_score = score(vector)
      # Save the vector.
      final_vectors.append({
                              'vector':vector,
                              'score':vector_score
                          })
                          
  # The vector having the best score among final vectors is recommended.
  # Sort final_vectors by score DESC.
  final_vectors = sorted(deepcopy(final_vectors), key=lambda dct: dct['score'], reverse=True)
  # Get the best score.
  best_score = final_vectors[0]['score']
  
  # Prints
  print("-----------------------------------")
  print("----------- AFTER GREEDY ----------")
  print("Best Vector Score = " + str(round(best_score,1)))
  print("-----------------------------------")
  
  sys.exit()
  """
  
  """
  # DOES NOT WORK
  # -----------------
  # MULTI MERGE
  # -----------------
  merged_vectors = []
  # Create all couples (combinations of r=2 elements) possible from candidate_vectors.
  r = 2
  couples = list(combinations(candidate_vectors, r))
  # Create a merged vector from each couple. A merged vector is a merge of all 1-digits from the two vectors of the couple.
  for k in range(len(couples)):
    
    # Create the merged vector.
    v1 = deepcopy(couples[k][0]['vector'])
    v2 = deepcopy(couples[k][1]['vector'])
    v_merged = v1
    for d in range(len(v2)):
      if v2[d] == 1:
        v_merged[d] = 1
    
    # Normalise the merged vector.
    v_merged = normalise(deepcopy(v_merged))
    
    # Calculate the merged vector score.
    v_merged_score = score(v_merged)

    # Save the merged vector.
    merged_vectors.append({
                            'vector':v_merged,
                            'score':v_merged_score
                          })
    
  # In fine, we have as many merged vectors as couples.
  # The merged vector having the best score is recommended.
  # Sort merged_vectors by score DESC.
  merged_vectors = sorted(deepcopy(merged_vectors), key=lambda dct: dct['score'], reverse=True)
  # Get the best score.
  best_score = merged_vectors[0]['score']   

  # Prints
  print("-----------------------------------")
  print("Best Merged Vector Score = " + str(best_score))
  print("-----------------------------------")
  
  sys.exit()
  """
  
  
  




  














