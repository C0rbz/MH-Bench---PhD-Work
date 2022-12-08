# These algorithms have been abandoned because they are too simple or already existing in the state of the art.
# I have modified and/or merged them in other forms in functions.py.



# Extossom
# Extossom = Extirpate + Blossom
# General Principle : 
# Start with a vector containing x 1-digits randomly determined.
# On each iteration : 
  # we extirpate the x best 1-digits in vector.
  # we create a new vector with the x 1-digits extirpated filled with other random 1-digits respecting the max number of 1-digits.
  # we keep the new vector if it is better.
# Keep the best vector through iterations.
def extossom():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  strongest_vector = []
  max_score = 0
  iteration_score_evolution = []
  max_score_evolution = []
  available_values = [0, 0, 0, 0, 0, 1] # x% luck to do something
  one_digits_indexes = []
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0

  # The initial vector containing 1-digits randomly determined.
  vector = generate_vector()
  
  # Saving the 1-digits indexes of the initial vector into one_digits_indexes.
  for k in range(len(vector)):   
    if vector[k] == 1:  
      one_digits_indexes.append(k)
  
  # Adding to top_vectors.    
  vector_score = score(vector)
  add_to_top_vectors(vector, vector_score)
  # Adding to all_vectors.    
  all_vectors.append(vector)

  # PARAMETERS
  extirpate_number = 8 # Amount of 1-digits extirpated to create a new vector. 
 
  # ITERATIONS
  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations
  
    # Memory of vector
    vector_memory = deepcopy(vector)
    one_digits_indexes_memory = deepcopy(one_digits_indexes)
    
    # Extirpate the x best 1-digits in vector.
    scores_1_digits = []
    for k in range(0, len(one_digits_indexes)):
      score_temp = resource_score(one_digits_indexes[k])
      scores_1_digits.append(score_temp)
    # Merge of one_digits_indexes and scores_1_digits into best_1_digits.
    best_1_digits = []
    for k in range(0, len(scores_1_digits)):
      best_1_digits.append({'index':one_digits_indexes[k], 'score':scores_1_digits[k]})
    # Sort best_1_digits by score DESC.
    best_1_digits = sorted(best_1_digits, key=lambda dct: dct['score'], reverse=True)
    
    # Create a new vector with the first extirpate_number elements from best_1_digits.
    vector = []
    one_digits_indexes = []
    for m in range(len(df)):
      vector.append(0)
    count_global_digits_1 = 0 
    # Add the first extirpate_number elements from best_1_digits.
    for k in range(0, extirpate_number):
      vector[best_1_digits[k]['index']] = 1
      one_digits_indexes.append(best_1_digits[k]['index'])
      count_global_digits_1 += 1
    # Complete with random 1-digits.
    while count_global_digits_1 < one_digits_needed: 
      rand = random.randint(0, len(vector)-1)
      if vector[rand] == 0:
        count_global_digits_1 += 1
        vector[rand] = 1
        one_digits_indexes.append(rand)

    # Adding to top_vectors.    
    vector_score = score(vector)
    add_to_top_vectors(vector, vector_score)
    # Adding to all_vectors.    
    all_vectors.append(vector)

    # EVALUATIONS - ITERATION END 
    iteration_score = vector_score
    iteration_score_evolution.append(iteration_score)
    
    if iteration_score > max_score:
      max_score = iteration_score
      strongest_vector = deepcopy(vector)
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:
      # Keep the memory of former vector version (because it is better) for next iteration. We do this because vector didn't improve during this iteration.
      vector = deepcopy(vector_memory)
      one_digits_indexes = deepcopy(one_digits_indexes_memory)
      # convergence management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score

    max_score_evolution.append(max_score)

  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution)

  if convergence_score == 0: # if convergence not reached, convergence is considered to be the max_score.
      convergence_iteration = max_score_iteration
      convergence_score = max_score
  return [
              convergence_iteration, 
              convergence_score, 
              max_score_iteration, 
              max_score, 
              res[0], 
              res[1],
              res[2],
              res[3],
              res[4],
            res[5],
            res[6],
            res[7]
          ]



# Weak Boom
# General Principle : 
# Start with a vector containing x 1-digits randomly determined.
# Change the weakest 1-digit into a 0-digit and a 0-digit into a 1-digit randomly on each iteration.
# Keep the best vector through iterations.
def weak_boom():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  strongest_vector = []
  max_score = 0
  iteration_score_evolution = []
  max_score_evolution = []
  available_values = [0, 0, 0, 0, 0, 1] # x% luck to do something
  one_digits_indexes = []
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0


  # The initial vector containing 1-digits randomly determined.
  vector = generate_vector()
      
  # Saving the 1-digits indexes of the initial vector into one_digits_indexes.
  for k in range(len(vector)):   
    if vector[k] == 1:  
      one_digits_indexes.append(k) 

  # Adding to top_vectors.    
  vector_score = score(vector)
  add_to_top_vectors(vector, vector_score)
  # Adding to all_vectors.    
  all_vectors.append(vector)
 
  # ITERATIONS
  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations
  
    # Memory of vector
    vector_memory = deepcopy(vector)
    one_digits_indexes_memory = deepcopy(one_digits_indexes)
    
    # DOWNGRADE - The weakest 1-digit become a 0-digit
    weakest_index = 0
    weakest_score = 100
    # Search for the weakest 1-digit.
    for k in range(0, len(one_digits_indexes)):
      score_k = resource_score(one_digits_indexes[k])
      if score_k < weakest_score:
        weakest_score = score_k
        weakest_index = one_digits_indexes[k]
    vector[weakest_index] = 0
    one_digits_indexes.remove(weakest_index)
    
    # UPGRADE - A random 0-digit become a 1-digit
    rand = random.randint(0, len(vector)-1)
    while rand in one_digits_indexes:
      rand = random.randint(0, len(vector)-1)
    vector[rand] = 1
    one_digits_indexes.append(rand)

    # Adding to top_vectors.    
    vector_score = score(vector)
    add_to_top_vectors(vector, vector_score)
    # Adding to all_vectors.    
    all_vectors.append(vector)

    # EVALUATIONS - ITERATION END 
    iteration_score = vector_score
    iteration_score_evolution.append(iteration_score)
    
    if iteration_score > max_score:
      max_score = iteration_score
      strongest_vector = deepcopy(vector)
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:
      # Keep the memory of former vector version (because it is better) for next iteration. We do this because vector didn't improve during this iteration.
      vector = deepcopy(vector_memory)
      one_digits_indexes = deepcopy(one_digits_indexes_memory)
      # convergence management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score

    max_score_evolution.append(max_score)
  
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution)

  if convergence_score == 0: # if convergence not reached, convergence is considered to be the max_score.
      convergence_iteration = max_score_iteration
      convergence_score = max_score
  return [
              convergence_iteration, 
              convergence_score, 
              max_score_iteration, 
              max_score, 
              res[0], 
              res[1],
              res[2],
              res[3],
              res[4],
            res[5],
            res[6],
            res[7]
          ]



# Tempo Without Frequency
# Just to check if the frequency really plays a role into Tempo algorithm.
# Experimentations show that :
  # Execution time is ultra fast like in Tempo.
  # Results are always better than all other algorithms like in Tempo.
  # The frequency plays no role. The simplicity of having only one mutation on each iteration is enough to get strong results. As a consequence, we can consider that for the moment this algorithm outperform Tempo but we have to keep in mind that in specific contexts, the presence of the frequency could play an interesting role.
def tempo_without_frequency():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  strongest_vector = []
  max_score = 0
  iteration_score_evolution = []
  max_score_evolution = []
  available_values = [0, 0, 0, 0, 0, 1] # x% luck to do something
  one_digits_indexes = []
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0


  # The initial vector containing 1-digits randomly determined.
  vector = generate_vector()
      
  # Saving the 1-digits indexes of the initial vector into one_digits_indexes.
  for k in range(len(vector)):   
    if vector[k] == 1:  
      one_digits_indexes.append(k) 

  # Adding to top_vectors.    
  vector_score = score(vector)
  add_to_top_vectors(vector, vector_score)
  # Adding to all_vectors.    
  all_vectors.append(vector)
    

  # ITERATIONS
  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations
  
    # Memory of vector
    vector_memory = deepcopy(vector)
    one_digits_indexes_memory = deepcopy(one_digits_indexes)
    
    # DOWNGRADE - A 1-digit become a 0-digit
    rand = random.choice(one_digits_indexes)
    vector[rand] = 0
    one_digits_indexes.remove(rand)
    
    # UPGRADE - A 0-digit become a 1-digit
    rand = random.randint(0, len(vector)-1)
    while rand in one_digits_indexes:
      rand = random.randint(0, len(vector)-1)
    vector[rand] = 1
    one_digits_indexes.append(rand)

    # Adding to top_vectors.    
    vector_score = score(vector)
    add_to_top_vectors(vector, vector_score)
    # Adding to all_vectors.    
    all_vectors.append(vector)


    # EVALUATIONS - ITERATION END 
    iteration_score = vector_score
    iteration_score_evolution.append(iteration_score)
    
    if iteration_score > max_score:
      max_score = iteration_score
      strongest_vector = deepcopy(vector)
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:
      # Keep the memory of former vector version (because it is better) for next iteration. We do this because vector didn't improve during this iteration.
      vector = deepcopy(vector_memory)
      one_digits_indexes = deepcopy(one_digits_indexes_memory)
      # convergence management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score

    max_score_evolution.append(max_score)
  
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution)

  if convergence_score == 0: # if convergence not reached, convergence is considered to be the max_score.
      convergence_iteration = max_score_iteration
      convergence_score = max_score
  return [
              convergence_iteration, 
              convergence_score, 
              max_score_iteration, 
              max_score, 
              res[0], 
              res[1],
              res[2],
              res[3],
              res[4],
            res[5],
            res[6],
            res[7]
          ]



# Tempo
# General Principle :
# Start with a vector containing x 1-digits randomly determined.
# Change a 1-digit into a 0-digit and a 0-digit into a 1-digit on each iteration.
# Keep the best vector through iterations.
# Particularity : A frequency is given as parameter. When a random number is picked, we make sure to pick it at a specific moment corresponding to a beat of the given frequency. This is equivalent to say that random numbers are picked on times which are multiples of 1/frequency seconds.
# Experimentations show that :
  # Changing the frequency only impact execution time. The higher the frequency the faster the execution time.
  # Execution time is ultra fast with appropriate frequency.
  # Results are always better than all other algorithms. Why ? => see algorithm tempo_without_frequency.
def tempo():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  strongest_vector = []
  max_score = 0
  iteration_score_evolution = []
  max_score_evolution = []
  available_values = [0, 0, 0, 0, 0, 1] # x% luck to do something
  one_digits_indexes = []
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0
  
  # Time reference for frequency.
  ref_time = time.perf_counter()
  
  # PARAMETERS
  frequency = 1000 # Frequency we respect when random numbers are picked.
  error_gap = 0.001 # Minimum rest allowed in while modulos. See below.


  # The initial vector containing 1-digits randomly determined.
  vector = generate_vector()
      
  # Saving the 1-digits indexes of the initial vector into one_digits_indexes.
  for k in range(len(vector)):   
    if vector[k] == 1:  
      one_digits_indexes.append(k) 

  # Adding to top_vectors.    
  vector_score = score(vector)
  add_to_top_vectors(vector, vector_score)
  # Adding to all_vectors.    
  all_vectors.append(vector)
    

  # ITERATIONS
  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations
  
    # Memory of vector
    vector_memory = deepcopy(vector)
    one_digits_indexes_memory = deepcopy(one_digits_indexes)
    
    time_now = time.perf_counter()
    # While we are not on a beat of given frequency.
    while (time_now-ref_time)%(1/frequency) > error_gap:
      time_now = time.perf_counter()
    
    # DOWNGRADE - A 1-digit become a 0-digit
    rand = random.choice(one_digits_indexes)
    vector[rand] = 0
    one_digits_indexes.remove(rand)

    time_now = time.perf_counter()
    # While we are not on a beat of given frequency.
    while (time_now-ref_time)%(1/frequency) > error_gap:
      time_now = time.perf_counter()
      
    # UPGRADE - A 0-digit become a 1-digit
    rand = random.randint(0, len(vector)-1)
    while rand in one_digits_indexes:
      time_now = time.perf_counter()
      # While we are not on a beat of given frequency.
      while (time_now-ref_time)%(1/frequency) > error_gap:
        time_now = time.perf_counter()
      rand = random.randint(0, len(vector)-1)
    vector[rand] = 1
    one_digits_indexes.append(rand)

    # Adding to top_vectors.    
    vector_score = score(vector)
    add_to_top_vectors(vector, vector_score)
    # Adding to all_vectors.    
    all_vectors.append(vector)


    # EVALUATIONS - ITERATION END 
    iteration_score = vector_score
    iteration_score_evolution.append(iteration_score)
    
    if iteration_score > max_score:
      max_score = iteration_score
      strongest_vector = deepcopy(vector)
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:
      # Keep the memory of former vector version (because it is better) for next iteration. We do this because vector didn't improve during this iteration.
      vector = deepcopy(vector_memory)
      one_digits_indexes = deepcopy(one_digits_indexes_memory)
      # convergence management
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score

    max_score_evolution.append(max_score)
  
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution)

  if convergence_score == 0: # if convergence not reached, convergence is considered to be the max_score.
      convergence_iteration = max_score_iteration
      convergence_score = max_score
  return [
              convergence_iteration, 
              convergence_score, 
              max_score_iteration, 
              max_score, 
              res[0], 
              res[1],
              res[2],
              res[3],
              res[4],
            res[5],
            res[6],
            res[7]
          ]



# Brawl
# General Principle : 
# A 5 brawlers pool. A contender_brawler confront them and take the place of one of them if it win.
# A contender_brawler is created by mutating the best brawler known. 
# Through iterations, most powerfull brawlers stay in the pool.
# We recommend the resources associated with the most powerfull brawler.
def brawl():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  
  # Function to create a contender_brawler (mutated best_brawler).
  def contender_brawler(best_brawler_copy):
    contender_brawler = best_brawler_copy
    # One 1-digit become a 0-digit and one 0-digit become a 1-digit.
    d0_to_d1 = 0
    d1_to_d0 = 0
    while d0_to_d1 < 1:
      index = random.randint(0, len(contender_brawler)-1)
      if contender_brawler[index] == 0:
        contender_brawler[index] = 1
        d0_to_d1 += 1
     
    available_values = [0, 0, 0, 0, 1]     
    while d1_to_d0 < 1: 
      for k in range(0, len(contender_brawler)):
        if d1_to_d0 < 1:
          digit = random.choice(available_values)
          if digit == 1 and contender_brawler[k] == 1:
            contender_brawler[k] = 0
            d1_to_d0 += 1
    return contender_brawler


  best_brawler = []
  max_score = 0
  mean_score_evolution = []
  iteration_score_evolution = []
  max_score_evolution = []
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0

  # Initial Brawlers Pool
  # A brawler contains 0-digits and 1-digits. A digit indicates the recommendation or not of a pair action-resource (observing the same index in df). For example, a brawler containing 1001100101101... means that the first action-resource in df is recommended (because first digit is a 1-digit) but not the second one (because second digit is a 0-digit), and so on.
  # We initialize brawlers with random digits. Each brawler contains as many digits as pairs action-resource in df.
  brawlers = []
  for k in range(5): # 5 brawlers pool
    vector = generate_vector()
    brawlers.append({
                      'vector':vector,
                      'score':score(vector)
                    })
  
  # Adding to top_vectors and all_vectors.   
  for k in range(len(brawlers)):
    add_to_top_vectors(brawlers[k]['vector'], brawlers[k]['score'])   
    all_vectors.append(brawlers[k]['vector'])


  round_number = algorithm_iterations
  for i in range(round_number): # x rounds
    """
    print("")
    print("Round N° " + str(i+1))
    print("")
    """
    # If it is the 1st iteration.
    if i == 0:
      # Sort the pool regarding scores ASC.
      brawlers = sorted(deepcopy(brawlers), key=lambda dct: dct['score'])
      # Save info about the best_brawler.
      brawlers_number = len(brawlers)
      max_score = brawlers[brawlers_number-1]['score']
      best_brawler = deepcopy(brawlers[brawlers_number-1]['vector'])
      
    # On each round, a new brawler enters the pool. Its digits are determined following the mutation of the best_brawler known so far.
    new_b_vector = contender_brawler(deepcopy(best_brawler))
    score_new_b = score(new_b_vector)
    new_b = {
              'vector':new_b_vector,
              'score':score_new_b
            }

    # We compare new_b to the weakest brawler in the pool (index 0 of brawlers).
    if new_b['score'] > brawlers[0]['score']:
      brawlers[0] = deepcopy(new_b)
      # We sort the pool again by score ASC because it has changed.
      brawlers = sorted(deepcopy(brawlers), key=lambda dct: dct['score'])

    # Adding to top_vectors.    
    add_to_top_vectors(new_b['vector'], new_b['score'])
    # Adding to all_vectors.    
    all_vectors.append(new_b['vector'])
    

    # EVALUATIONS - ROUND END       
    over_max_score = False
    score_max = brawlers[len(brawlers)-1]['score']
    if score_max > max_score:
      over_max_score = True
      max_score = score_max
      best_brawler = deepcopy(brawlers[len(brawlers)-1]['vector'])
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
          
    if over_max_score is False:
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score
    
    iteration_score = new_b['score']
    iteration_score_evolution.append(iteration_score)
    
    max_score_evolution.append(max_score)
    
    scores_brawlers = []
    for k in range(0, len(brawlers)):
      scores_brawlers.append(brawlers[k]['score'])
    mean_score_evolution.append(
                                  mean(scores_brawlers)
                                )
  
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution)

  if convergence_score == 0: # if convergence not reached, convergence is considered to be max_score.
    convergence_iteration = max_score_iteration
    convergence_score = max_score
  return [
              convergence_iteration, 
              convergence_score, 
              max_score_iteration, 
              max_score, 
              res[0], 
              res[1],
              res[2],
              res[3],
              res[4],
            res[5],
            res[6],
            res[7]
          ]




# OLD
# Guide Exploration & Exploitation (E&E).
# Function to check if the balance between E&E is good regarding thresholds on a global similarity percentage. If the balance is good, the function returns the population without modifications, else the function returns the population modified according to the E&E situation. Before being returned, the population is normalised.
# The population parameter must contain at least two vectors.
def OLD_guide_explo(population):

  # Number of vectors in the population.
  pop_size = len(population)  
  
  # Get a vector representing how many times each digit is a 1-digit in the entire population.
  summed_vectors = sum(population, axis=0)
  
  # Calculate the global indicator of similarity between population vectors.
  # To do that, we divide each digit's number of appearance by the number of vectors in the population. At the end, we do a mean of all the values obtained.
  # Please note that the maximum value of e_e_indicator is 1. So, e_e_indicator is a percentage. This percentage represents the mean number of presence, of each digit, in all the population vectors. So you can read this percentage as "Each digit is present in x% of vectors" or as "Vectors' similarity is x%".
  digit_values = []
  for k in range(len(summed_vectors)):
    if summed_vectors[k] > 0:
      digit_values.append(summed_vectors[k]/pop_size) 
  e_e_indicator = mean(digit_values)
  
  # The more e_e_indicator is high, the more the vectors are similar corresponding to a behaviour too much centered on exploitation.
  # The more e_e_indicator is low, the more the vectors are different corresponding to a behaviour too much centered on exploration. 
  # So, we just have to put the limits to know when to consider if we are in the first case or in the second one.
  e_e_min = 0.5
  e_e_max = 0.9
  
 
  # If the behaviour is too much centered on exploration.
  # We need to force exploitation.
  if e_e_indicator < e_e_min: 
    
    #print("===> E X P L O I T - E&E= " + str(round(e_e_indicator, 2)), end="\r")
    print("===> E X P L O I T - E&E= " + str(round(e_e_indicator, 2)))
      
    # Get the 1-digits indexes among all vectors. This means that we get digits already exploited.
    one_digits_indexes = []
    for k in range(len(summed_vectors)):
      if summed_vectors[k] > 0:
        one_digits_indexes.append(k)

    while e_e_indicator < e_e_min:
      
      # In each vector of population, random 0-digits becomes 1-digits if these digits have already been exploited.
      for k in range(len(population)):
        # Copy one_digits_indexes to be able to remove already picked indexes from the copy.
        copy_one_digits_indexes = deepcopy(one_digits_indexes)
        done = False
        # Do the following until finding a 0-digit that can become a 1-digit according to the index picked, and while there is still at least one index to pick and while constraints are respected.
        while done is False and len(copy_one_digits_indexes) > 0:
          rand = random.choice(copy_one_digits_indexes)
          if population[k][rand] == 0:
            population[k][rand] = 1
            if constraints_max_values_check(population[k]) is False:
              done = True # End while loop.
          else:
            copy_one_digits_indexes.remove(rand)
            
      # Update e_e_indicator.
      summed_vectors = sum(population, axis=0)
      digit_values = []
      for k in range(len(summed_vectors)):
        if summed_vectors[k] > 0:
          digit_values.append(summed_vectors[k]/pop_size) 
      e_e_indicator = mean(digit_values)
  
  # Else if the behaviour is too much centered on exploitation.
  # We need to force exploration.
  elif e_e_indicator > e_e_max:
    
    #print("===> E X P L O R E - E&E= " + str(round(e_e_indicator, 2)), end="\r")
    print("===> E X P L O R E - E&E= " + str(round(e_e_indicator, 2)))
      
    # Get the 0-digits indexes among all vectors. This means that we get digits not already exploited. We want to explore them by exploiting them.
    v_zero_digits_indexes = []
    for k in range(len(summed_vectors)):
      if summed_vectors[k] == 0:
        v_zero_digits_indexes.append(k)
        
    while e_e_indicator > e_e_max:

      """
      # Remove randomly 25% of the vectors from population.
      print("==================================> REMOVE 25%")
      remove_quantity = int(0.25 * pop_size)
      if remove_quantity == 0:
        remove_quantity = 1
      while remove_quantity > 0:
        rand = random.randint(0, len(population)-1)
        population.pop(rand)
        remove_quantity -= 1
      # Add in the population as many fresh generated vectors as the quantity removed.
      for k in range(remove_quantity):
        population.append(generate_vector())
      """
      
      # In each vector of population, random 0-digits, not exploited yet among all vectors, becomes 1-digits.
      for k in range(len(population)):
        ok = True
        # Do the following until finding a 0-digit that can become a 1-digit according to the index picked, and while there is still at least one index to pick and while constraints are respected.
        while ok is True and len(v_zero_digits_indexes) > 0:
          # A random 0-digit becomes a 1-digit.
          rand = random.choice(v_zero_digits_indexes)
          population[k][rand] = 1
          v_zero_digits_indexes.remove(rand)
          if constraints_max_values_check(population[k]) is False:
            ok = False # End while loop.
            #population[k][rand] = 0 # CANCEL LAST MODIFICATION
            #v_zero_digits_indexes.append(rand)
            
      # Update e_e_indicator.
      summed_vectors = sum(population, axis=0)
      digit_values = []
      for k in range(len(summed_vectors)):
        if summed_vectors[k] > 0:
          digit_values.append(summed_vectors[k]/pop_size) 
      e_e_indicator = mean(digit_values)
    
  # Else, all is good so we don't modify the population.
  else:
    print("ALL GOOD - E&E= " + str(round(e_e_indicator, 2)), end="\r")
    #print("ALL GOOD - E&E= " + str(round(e_e_indicator, 2)))
  
  # Normalise the population.
  for k in range(len(population)):
    population[k] = normalise(deepcopy(population[k]))
  
  return population




# Guide Explo With Golden Number.
# Function to guide exploration and exploitation by representing the search space of each vector as a geometrical shape and by judging the fitness of this shape toward the golden number.
def guide_explo_with_golden_number(population):
  # Initialise g_pop
  g_pop = []
  # For each vector in the population
  for k in range(len(population)):
    # Declare the rests array
    rests = []
    # For each 1-digit in the vector
    for d in range(len(population[k])):
      if population[k][d] == 1:
        # Calculate the number of indexes that separates it from the next 1-digit.
        # To find the next 1-digit, the vector is considered as a circle. So, if the end of the vector is reached, we continue iterating from its beginning.
        index_span = 0
        # Set d2
        if d == (len(population[k])-1):
          d2 = 0
        else:
          d2 = d+1
        # Loop
        found = False
        while found is False:
          index_span += 1
          # End loop condition : next 1-digit found
          if population[k][d2] == 1:
            found = True
          # Update d2 for next iteration
          if d2 == (len(population[k])-1):
            d2 = 0
          else:
            d2 += 1
        # Calculate the modulo of the final index_span with the golden number.
        rest = index_span%((1+sqrt(5))/2)
        # Add the rest obtained into the rests array
        rests.append(rest)
    # Calculate the mean of the rests array.
    mean_rest = mean(rests)
    # Add to g_pop the tuple (vector, mean_rest)
    g_pop.append({
                    'vector':population[k],
                    'mean_rest':mean_rest
                })
  
  # Sort g_pop by mean_rest ASC
  # We sort by order ASC because vectors being the least compliant with the golden number have the biggest mean_rests and so are put to the end of g_pop.
  g_pop = sorted(deepcopy(g_pop), key=lambda dct: dct['mean_rest'])

  """
  # Remove the lasts 25% vectors of g_pop : the worst ones regarding geometrical compliance with the golden number.
  remove_quantity = int(0.25 * len(g_pop))
  if remove_quantity < 1:
    remove_quantity = 1
  # Remove
  for k in range(remove_quantity):
    g_pop.pop()
  # Add in g_pop as many fresh generated vectors as the quantity removed.
  for k in range(remove_quantity):
    g_pop.append({
                    'vector':generate_vector()
                })
  """
  
  # Reset population and put each vector from g_pop into population.
  population = []
  for k in range(len(g_pop)):
    population.append(g_pop[k]['vector'])

  # Enrich the lasts 25% vectors of g_pop : the worst ones regarding geometrical compliance with the golden number.
  # In each vector of the lasts 25% vectors, a 0-digit becomes a 1-digit while constraints are respected.
  quantity_to_enrich = int(0.25 * len(g_pop))
  if quantity_to_enrich < 1:
    quantity_to_enrich = 1
  start = len(population) - 1 - quantity_to_enrich
  end = len(population)
  for k in range(start, end):
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
    
  # Return population
  return population




# Function to do E&E variations without the guiding e_e_indicator.
# Allows to compare with the guide_explo() function in order to see if the guiding process is really usefull compared to straight added variations.
# Before being returned, the population is normalised.
# The population parameter must contain at least two vectors.
def explo_variations_without_guiding(population):

  # Number of vectors in the population.
  pop_size = len(population)  
  
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























