# This file contains implementations of state of the art evolutionary algorithms.
# They are used in experimentations for comparison with our proposed algorithms.
# NOTA : Our proposed algorithms can be found in functions.py.


""" State of the art algorithms - Multi-objective """
# NSGA-III - 2014     => ABANDONED (computationally expensive)
# MOMBI2 - 2015       => ABANDONED (no detailed presentation)
# SPEA2+SDE - 2014    => ABANDONED (computationally expensive)
# MOEA/D-DD - 2013    => ABANDONED (no detailed presentation)
# WSK - 2019          => ABANDONED (computationally expensive)
# LIBEA - 2019        => ABANDONED (computationally expensive)
# RSEA - 2019         => ABANDONED (computationally expensive)
# SPEA/R - 2017       => ABANDONED (computationally expensive)
# SGEA - 2017         => ABANDONED (computationally expensive)
# VaEA - 2017         => ABANDONED (computationally expensive)
# θ-DEA - 2016        => ABANDONED (computationally expensive)
# HypE - 2008         => ABANDONED (computationally expensive)
# MOEA/D - 2007       => ABANDONED (computationally expensive)
# GANGSTER - 2016     => ABANDONED (it is a negotiator, not our approach)
# GA-MANS-GRS - 2018  => ABANDONED (it is a negotiator, not our approach)

""" State of the art algorithms - Others """
# BDDEA-LDG - 2020    => ABANDONED (based on data generation, not our approach)
# DDEA-PES - 2020     => ABANDONED (based on data generation, not our approach)
# CAL-SAPSO           => ABANDONED (based on data generation, not our approach)
# SA-COSO             => ABANDONED (based on data generation, not our approach)
# GPEME               => ABANDONED (based on data generation, not our approach)
# DDEA-SE             => ABANDONED (based on data generation, not our approach)
# MGP-SLPSO           => ABANDONED (based on data generation, not our approach)
# EAS-SM3             => ABANDONED (based on data generation, not our approach)
# EAS-SM5             => ABANDONED (based on data generation, not our approach)
# EAS-SM12            => ABANDONED (based on data generation, not our approach)
# HEA-ACT - 2009      => ABANDONED (computationally expensive)




# DCEA - 2012
def dcea():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []

  # Strongest vector
  v_strong = []
  # Score of the strongest vector
  max_score = 0

  iteration_score_evolution = []
  max_score_evolution = []

  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of max_score
  max_score_iteration = 0

  population = []

  # PARAMETERS
  population_size = 10
  pc = 0.8 # Crossover probability
  pm = 0.05 # Mutation probability


  # INITIALIZATION
  # Generate the initial population.
  for i in range(0, population_size):
    # Create a vector.
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

  # Sort population by score DESC.
  population = sorted(deepcopy(population), key=lambda dct: dct['score'], reverse=True)


  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations
    
    #print("Iteration N° " + str(i+1))

    offspring_population = []

    # Selection process using tournament selection + Pairing process of selected vectors.
    # In general, the solutions with the greatest fitness values have more chances of being selected.
    # Tournament selection explanation :
    # Two solutions are randomly selected from the current population.
    # Then, the fitness values of those solutions are compared.
    # The better one (i.e., the solution with the best fitness value) is selected and becomes the first member of the pair which is being designed. The other solution is returned to the population.
    # This operation is repeated so as to obtain the second member of the pair. A pair of solutions is thus built out of the current population.
    # The method described in the previous paragraph is applied M/2 times to the current population to obtain M/2 pairs, considering M as the size of the population.
    # Let's do it :
    parents_pairs = []
    for k in range(int(population_size/2)):
      pair = []
      for t in range(0, 2): # We get two vectors to build a pair.
        rand_index_1 = random.randint(0, population_size-1)
        rand_index_2 = random.randint(0, population_size-1)
        if population[rand_index_1]['score'] >= population[rand_index_2]['score']:
          pair.append(deepcopy(population[rand_index_1]))
        else:
          pair.append(deepcopy(population[rand_index_2]))
      parents_pairs.append(deepcopy(pair))


    # Crossover process applied to each pair with probability "pc". 
    # Operator used : order crossover. This operator is described below.
    # The operator is applied to the parent 1 and parent 2 of the pair, which generates two new solutions, offspring 1 and offspring 2.
    # The operator defines two random crossover points k1 and k2, considering 
    # 1 <= k1 < k2 < n, and n is equal to the length of the parent solutions. Then, to define offspring 1, the operation is as follows. 
    # Firstly, the elements in the segment [k1,k2] of parent 1 are copied, in the same order, in the segment [k1,k2] of offspring 1. 
    # Then, the operator copies the elements not included in offspring 1 in the empty positions of this solution. The elements not included in offspring 1 are copied taking into account the order in which they appear in parent 2. Specifically, starting from the second crossover point in parent 2, the operator copies the elements not included in the order in which they appear in parent 2. When the operator reaches the last element on parent 2 list, the process continues from the first position on that list.
    # The generation of offspring 2 is similar to the generation of offspring 1. However, the roles of the parents are inverted to generate offspring 2.
    # Let's do it :
    for k in range(len(parents_pairs)):
      offspring_1 = deepcopy(parents_pairs[k][1]) # parent_2
      offspring_2 = deepcopy(parents_pairs[k][0]) # parent_1

      # Do we have to do the crossover according to "pc" ?
      picked_number = (random.randint(1, 100))/100
      if picked_number <= pc:
        # Do the crossover.
        k1 = random.randint(0, len(df)-3)
        k2 = random.randint(k1+1, len(df)-2)
        
        for d in range(k1, k2+1):
          offspring_1['vector'][d] = deepcopy(parents_pairs[k][0]['vector'])[d]
          offspring_2['vector'][d] = deepcopy(parents_pairs[k][1]['vector'])[d]
        
        # Assure that offspring_1 and offspring_2 respect constraints.
        offspring_1['vector'] = normalise(deepcopy(offspring_1['vector']))
        offspring_2['vector'] = normalise(deepcopy(offspring_2['vector']))

        # Calculate scores of offspring_1 and offspring_2.
        offspring_1['score'] = score(deepcopy(offspring_1['vector']))
        offspring_2['score'] = score(deepcopy(offspring_2['vector']))

      # Feed offspring_population
      offspring_population.append(deepcopy(offspring_1))
      offspring_population.append(deepcopy(offspring_2))
      
      
    # Mutation process applied according to probability "pm".
    # Operator used : swap mutation. This operator is described below.
    # This operator starts by randomly selecting two positions k1 and k2, considering 
    # 1 <= k1 < k2 <= n, and n equal to the length of the solution. 
    # The mutation operator swaps the elements in positions k1 and k2.
    # Let's do it :
    for k in range(len(offspring_population)):
      # Do we have to do the mutation according to "pm" ?
      picked_number = (random.randint(1, 100))/100
      if picked_number <= pm:
        # Do the mutation.
        k1 = random.randint(0, len(df)-2)
        k2 = random.randint(k1+1, len(df)-1)
        value_at_k1 = deepcopy(offspring_population[k]['vector'])[k1]
        value_at_k2 = deepcopy(offspring_population[k]['vector'])[k2]
        
        if value_at_k1 != value_at_k2: 
          # Swaps
          offspring_population[k]['vector'][k1] = value_at_k2
          offspring_population[k]['vector'][k2] = value_at_k1
          # Recalculate score.
          offspring_population[k]['score'] = score(deepcopy(offspring_population[k]['vector']))

    # Add to top_vectors and all_vectors.
    for k in range(len(offspring_population)):  
      add_to_top_vectors(deepcopy(offspring_population[k]['vector']), offspring_population[k]['score'])   
      all_vectors.append(deepcopy(offspring_population[k]['vector']))

    # Sort offspring_population by score DESC.
    offspring_population = sorted(deepcopy(offspring_population), key=lambda dct: dct['score'], reverse=True)

    # offspring_population is the population for next iteration.
    population = deepcopy(offspring_population)
    
    # Strongest vector (v_strong) and associated score (max_score)
    over_max_score = False
    if offspring_population[0]['score'] > max_score:
      over_max_score = True
      max_score = offspring_population[0]['score']
      v_strong = deepcopy(offspring_population[0]['vector'])
    
    iteration_score_evolution.append(offspring_population[0]['score'])
    
    if over_max_score is True:
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score
    
    max_score_evolution.append(max_score)
    
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution)

  if convergence_score == 0: # if convergence not yet reached, convergence is considered to be max_score.
    convergence_iteration = max_score_iteration
    convergence_score = max_score
  return [
              convergence_iteration, 
              convergence_score, 
              max_score_iteration, 
              max_score, 
              res
          ]





# DE/EDA - 2005
# NOTA : the word "point" in comments refers to a vector in our problem.
def de_eda():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []

  # Strongest vector
  v_strong = []
  # Score of the strongest vector
  max_score = 0

  iteration_score_evolution = []
  max_score_evolution = []

  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of max_score
  max_score_iteration = 0

  population = []

  # PARAMETERS
  population_size = 10

  # INITIALIZATION
  # Generate the initial population.
  for i in range(0, population_size):
    # Create a vector.
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

  # Sort population by score DESC.
  population = sorted(deepcopy(population), key=lambda dct: dct['score'], reverse=True)


  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations
    
    #print("Iteration N° " + str(i+1))
    
    # Do the DE/EDA offspring generation so as to generate an offspring population.
    offspring_population = []
    
    
    # Step 1.1: Selection. Select M promising solutions from Pop(t) to form the parent set Q(t) by a selection method (like the truncation selection).
    # Truncation selection : individuals are sorted according to their objective function values. Only the best individuals are selected as parents.
    # Let's do it :
    # Take the x best vectors from population in order to form the parents set.
    x = 4 # We choose to get 4 parents
    parents = []
    for k in range(0, x):
      parents.append(deepcopy(population[k]))
      
      
    # Step 1.2: Modelling. Build a probabilistic model p(x) based on the statistical information extracted from the solutions in Q(t).
    # p(x) can be a Gaussian distribution, a Gaussian mixture, a histogram, or a Gaussian model with diagonal covariance matrix (GM/DCM).
    # GM/DCM is used in our algorithm.
    # Initialisation of proba_model with 1 values.
    proba_model = []
    for k in range(len(df)):
      proba_model.append(1)
    for k in range(len(parents)):
      mu = mean(deepcopy(parents[k]['vector']))
      sigma = std(deepcopy(parents[k]['vector']))
      #normal_distribution = (1/sqrt(2*math.pi*sigma)) * math.exp(0.5*((parents[k]['vector']-mu)/sigma)**2)
      term_1 = 1/sqrt(2*math.pi*sigma)
      term_2 = [math.exp((((x-mu)/sigma)**2)*0.5) for x in deepcopy(parents[k]['vector'])]
      normal_distribution = [term_1*x for x in term_2]
      #proba_model *= normal_distribution
      proba_model = [a_i * b_i for a_i, b_i in zip(deepcopy(proba_model), deepcopy(normal_distribution))]      
      

    # STEP 2 : For each vector of the population : Generate a trial point u = (u1,u2, . . .,un) as follows:
    sigma = 0.5
    F = 0.5 # F is a given control parameter.
    for k in range(len(population)):
    
      # Choose stuff.
      # Choose a random vector having a score >= to population[k]['score'].
      v_sup = []
      found_sup = False
      while found_sup is False:
        rand = random.randint(0, len(population)-1)
        if population[rand]['score'] >= population[k]['score']:
          found_sup = True
          v_sup = deepcopy(population[rand])
      # Choose two other random vectors from the population.
      rand = random.randint(0, len(population)-1)
      v_random_1 = deepcopy(population[rand])
      rand = random.randint(0, len(population)-1)
      v_random_2 = deepcopy(population[rand])
      # Choose a subset S={j1,...,jm} of the index set {1,...,n}, while m < n and all ji mutually different.
      indexes_subset = []
      index_quantity = random.randint(0, len(population)-2)
      for n in range(0, index_quantity):
        added = False
        while added is False:
          random_index = random.randint(0, len(population)-1)
          if random_index not in indexes_subset: 
            indexes_subset.append(random_index)
            added = True
      
      rand = random.uniform(0, 1) # random number uniformly distributed within [0, 1]
      u = {
            'vector':[],
            'score':0
          }
      if(rand < sigma):
        #u['vector'] = (population[k]['vector']+v_sup)/2 
        #  + F*(v_sup-population[k]['vector']+v_random_1['vector']-v_random_2['vector'])
        term_1 = [(a_i + b_i)/2 for a_i, b_i in zip(deepcopy(population[k]['vector']), deepcopy(v_sup['vector']))]
        term_2 = [F*(a_i - b_i + c_i - d_i) for a_i, b_i, c_i, d_i in zip(deepcopy(v_sup['vector']), deepcopy(population[k]['vector']), deepcopy(v_random_1['vector']), deepcopy(v_random_2['vector']))]
        u['vector'] = [a_i + b_i for a_i, b_i in zip(deepcopy(term_1), deepcopy(term_2))]
      else:
        # u is sampled according to the constructed probabilistic model p(x).
        # Represents proba_model as tuples (index, proba_value).
        reorganised_proba_model = []
        for m in range(len(proba_model)):
          reorganised_proba_model.append({
                                            'index':m,
                                            'proba_value':deepcopy(proba_model[m])
                                        })
        # Sort reorganised_proba_model by proba_value DESC.
        reorganised_proba_model = sorted(deepcopy(reorganised_proba_model), key=lambda dct: dct['proba_value'], reverse=True)
        # Generate u, using best proba values from reorganised_proba_model.
        for d in range(len(df)):
          u['vector'].append(0)
        # Select random digits from the first quarter of reorganised_proba_model and change associated digits into 1-digits in u. Do it while u respects max_value constraints.
        indexes_already_chosen = []
        while constraints_max_values_check(u['vector']) is True:
          rand = random.randint(0, int(len(reorganised_proba_model)/4))
          if rand not in indexes_already_chosen:
            indexes_already_chosen.append(rand)
            u['vector'][reorganised_proba_model[rand]['index']] = 1
      
      
      # u treatments.
      # Reshape u['vector'] so that it contains only 0-digits and 1-digits.
      for d in range(len(u['vector'])):
        if u['vector'][d] > 0:
          u['vector'][d] = 1
        else:
          u['vector'][d] = 0
      # Assure that u respects constraints.
      u['vector'] = normalise(deepcopy(u['vector']))
      # Calculate the score of u.
      u['score'] = score(deepcopy(u['vector']))

      # STEP 3 : If f(u)>f(population[k]['vector']): set offspring_vector = u['vector'] else set offspring_vector = population[k]['vector']
      offspring_vector = []
      if u['score'] > population[k]['score']: 
        # offspring_vector become u
        offspring_vector = deepcopy(u) 
      else:
        # offspring_vector become population[k]
        offspring_vector = deepcopy(population[k])
      # Add offspring_vector to offspring_population.
      offspring_population.append(deepcopy(offspring_vector))

      # Add u to top_vectors and all_vectors.
      add_to_top_vectors(deepcopy(u['vector']), u['score'])   
      all_vectors.append(deepcopy(u['vector']))


    # Sort offspring_population by score DESC.
    offspring_population = sorted(deepcopy(offspring_population), key=lambda dct: dct['score'], reverse=True)

    # offspring_population is the population for next iteration.
    population = deepcopy(offspring_population)
    
    
    # Strongest vector (v_strong) and associated score (max_score)
    over_max_score = False
    if offspring_population[0]['score'] > max_score:
      over_max_score = True
      max_score = offspring_population[0]['score']
      v_strong = deepcopy(offspring_population[0]['vector'])
    
    iteration_score_evolution.append(offspring_population[0]['score'])
    
    if over_max_score is True:
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score
    
    max_score_evolution.append(max_score)
    
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution)

  if convergence_score == 0: # if convergence not yet reached, convergence is considered to be max_score.
    convergence_iteration = max_score_iteration
    convergence_score = max_score
  return [
              convergence_iteration, 
              convergence_score, 
              max_score_iteration, 
              max_score, 
              res
          ]


# EDA
# NOTA : the word "point" in comments refers to a vector in our problem.
def eda():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []

  # Strongest vector
  v_strong = []
  # Score of the strongest vector
  max_score = 0

  iteration_score_evolution = []
  max_score_evolution = []

  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of max_score
  max_score_iteration = 0

  population = []

  # PARAMETERS
  population_size = 10

  # INITIALIZATION
  # Generate the initial population.
  for i in range(0, population_size):
    # Create a vector.
    vector = generate_vector()
    
    vector_score = score(vector)
    # Add the vector to the population.
    population.append({
                        'vector':deepcopy(vector),
                        'score':vector_score
                      }) 
    # Add to top_vectors and all_vectors.
    add_to_top_vectors(deepcopy(vector), vector_score)   
    all_vectors.append(deepcopy(vector))
  
  # Sort population by score DESC.
  population = sorted(deepcopy(population), key=lambda dct: dct['score'], reverse=True)


  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations
    
    #print("Iteration N° " + str(i+1))

    # Step 1: Selection. Select M promising solutions from Pop(t) to form the parent set Q(t) by a selection method (like the truncation selection).
    # Truncation selection : individuals are sorted according to their objective function values. Only the best individuals are selected as parents.
    # Let's do it :
    # Take the x best vectors from population in order to form the parents set.
    x = 4 # We choose to get 4 parents
    parents = []
    for k in range(0, x):
      parents.append(deepcopy(population[k]))


    # Step 2: Modelling. Build a probabilistic model p(x) based on the statistical information extracted from the solutions in Q(t).
    # p(x) can be a Gaussian distribution, a Gaussian mixture, a histogram, or a Gaussian model with diagonal covariance matrix (GM/DCM).
    # GM/DCM is used in our algorithm.
    # Initialisation of proba_model with 1 values.
    proba_model = []
    for k in range(len(df)):
      proba_model.append(1)
    for k in range(len(parents)):
      mu = mean(deepcopy(parents[k]['vector']))
      sigma = std(deepcopy(parents[k]['vector']))
      #normal_distribution = (1/sqrt(2*math.pi*sigma)) * math.exp(0.5*((parents[k]['vector']-mu)/sigma)**2)
      term_1 = 1/sqrt(2*math.pi*sigma)
      term_2 = [math.exp((((x-mu)/sigma)**2)*0.5) for x in deepcopy(parents[k]['vector'])]
      normal_distribution = [term_1*x for x in term_2]
      #proba_model *= normal_distribution
      proba_model = [a_i * b_i for a_i, b_i in zip(deepcopy(proba_model), deepcopy(normal_distribution))]

    # Step 3: Sampling. Sample new solutions according to the constructed probabilistic model p(x).
    # Represents proba_model as tuples (index, proba_value).
    reorganised_proba_model = []
    for k in range(len(proba_model)):
      reorganised_proba_model.append({
                                        'index':k,
                                        'proba_value':deepcopy(proba_model[k])
                                    })
    # Sort reorganised_proba_model by proba_value DESC.
    reorganised_proba_model = sorted(deepcopy(reorganised_proba_model), key=lambda dct: dct['proba_value'], reverse=True)
    
    # Generate a number of new vectors equal to half of the population size, using best proba values from reorganised_proba_model.
    new_vectors = []
    for k in range(int(len(population)/2)):
      v = []
      for d in range(len(df)):
        v.append(0)
      # Select random digits from the first quarter of reorganised_proba_model and change associated digits into 1-digits in v. Do it while v respects max_value constraints.
      indexes_already_chosen = []
      while constraints_max_values_check(v) is True:
        rand = random.randint(0, int(len(reorganised_proba_model)/4))
        if rand not in indexes_already_chosen:
          indexes_already_chosen.append(rand)
          v[reorganised_proba_model[rand]['index']] = 1
      
      # Normalise
      v = normalise(deepcopy(v))
      
      score_v = score(deepcopy(v))
      new_vectors.append({
                            'vector':deepcopy(v),
                            'score':score_v
                        })
      
    # Step 4: Replacement. Fully or partly replace solutions in Pop(t) by the sampled new solutions to form a new population Pop(t + 1).
    # We replace vectors in population having the worse scores by the vectors of new_vectors.
    for k in range(len(new_vectors)):
      population.pop()
    for k in range(len(new_vectors)):
      population.append(deepcopy(new_vectors[k]))


    # Add vectors from new_vectors to top_vectors and all_vectors.
    for k in range(len(new_vectors)):
      add_to_top_vectors(deepcopy(new_vectors[k]['vector']), new_vectors[k]['score'])   
      all_vectors.append(deepcopy(new_vectors[k]['vector']))

    # Sort population by score DESC.
    population = sorted(deepcopy(population), key=lambda dct: dct['score'], reverse=True)


    # Strongest vector (v_strong) and associated score (max_score)
    over_max_score = False
    if population[0]['score'] > max_score:
      over_max_score = True
      max_score = population[0]['score']
      v_strong = deepcopy(population[0]['vector'])
    
    iteration_score_evolution.append(population[0]['score'])
    
    if over_max_score is True:
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score
    
    max_score_evolution.append(max_score)
    
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution)

  if convergence_score == 0: # if convergence not yet reached, convergence is considered to be max_score.
    convergence_iteration = max_score_iteration
    convergence_score = max_score
  return [
              convergence_iteration, 
              convergence_score, 
              max_score_iteration, 
              max_score, 
              res
          ]


# DE
# NOTA : the word "point" in comments refers to a vector in our problem.
def de():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []

  # Strongest vector
  v_strong = []
  # Score of the strongest vector
  max_score = 0

  iteration_score_evolution = []
  max_score_evolution = []

  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of max_score
  max_score_iteration = 0

  population = []

  # PARAMETERS
  population_size = 10

  # INITIALIZATION
  # Generate the initial population.
  for i in range(0, population_size):
    # Create a vector.
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


  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations
    
    #print("Iteration N° " + str(i+1))
    
    # Do the DE offspring generation for each vector of the population so as to generate an offspring population.
    offspring_population = []
    for k in range(len(population)):
      # STEP 1 : Choose stuff.
      # 1.1 : Choose a random vector having a score >= to population[k]['score'].
      v_sup = []
      found_sup = False
      while found_sup is False:
        rand = random.randint(0, len(population)-1)
        if population[rand]['score'] >= population[k]['score']:
          found_sup = True
          v_sup = deepcopy(population[rand])
      # 1.2 : Choose two other random vectors from the population.
      rand = random.randint(0, len(population)-1)
      v_random_1 = deepcopy(population[rand])
      rand = random.randint(0, len(population)-1)
      v_random_2 = deepcopy(population[rand])
      # 1.3 : Choose a subset S={j1,...,jm} of the index set {1,...,n}, while m < n and all ji mutually different.
      indexes_subset = []
      index_quantity = random.randint(0, len(population)-2)
      for n in range(0, index_quantity):
        added = False
        while added is False:
          random_index = random.randint(0, len(population)-1)
          if random_index not in indexes_subset: 
            indexes_subset.append(random_index)
            added = True
      
      # STEP 2 : Generate a trial point u = (u1,u2, . . .,un) as follows:
      # 2.1 : DE Mutation - Generate a temporary point z. 
      # F is a given control parameter.
      F = 0.5
      #z = (F+0.5)*v_sup['vector']
      #    +(F-0.5)*population[k]['vector']
      #    +F*(v_random_1['vector']-v_random_2['vector'])
      term_1 = [(F+0.5)*x for x in deepcopy(v_sup['vector'])]
      term_2 = [(F-0.5)*x for x in deepcopy(population[k]['vector'])]
      term_3_part = [a_i - b_i for a_i, b_i in zip(deepcopy(v_random_1['vector']), deepcopy(v_random_2['vector']))]
      term_3 = [F*x for x in term_3_part]
      z = [a_i + b_i + c_i for a_i, b_i, c_i in zip(deepcopy(term_1), deepcopy(term_2), deepcopy(term_3))]
      # 2.2: DE Crossover
      # u['vector'] is set as population[k]['vector'].
      # For j in indexes_subset, uj is chosen to be zj.
      u = deepcopy(population[k])
      for ind in indexes_subset:
        u['vector'][ind] = z[ind]

      # u treatments.
      # Reshape u['vector'] so that it contains only 0-digits and 1-digits.
      for d in range(len(u['vector'])):
        if u['vector'][d] > 0:
          u['vector'][d] = 1
        else:
          u['vector'][d] = 0
      # Assure that u respects constraints.
      u['vector'] = normalise(deepcopy(u['vector']))
      # Calculate the score of u.
      u['score'] = score(deepcopy(u['vector']))
     
      # STEP 3 : If f(u)>f(population[k]['vector']): set offspring_vector = u['vector'] else set offspring_vector = population[k]['vector']
      offspring_vector = []
      if u['score'] > population[k]['score']: 
        # offspring_vector become u
        offspring_vector = deepcopy(u)
      else:
        # offspring_vector become population[k]
        offspring_vector = deepcopy(population[k])

      # Add offspring_vector to offspring_population.
      offspring_population.append(deepcopy(offspring_vector))

      # Add u to top_vectors and all_vectors.
      add_to_top_vectors(deepcopy(u['vector']), u['score'])   
      all_vectors.append(deepcopy(u['vector']))

    # Sort offspring_population by score DESC.
    offspring_population = sorted(deepcopy(offspring_population), key=lambda dct: dct['score'], reverse=True)

    # offspring_population is the population for next iteration.
    population = deepcopy(offspring_population)
    
    
    # Strongest vector (v_strong) and associated score (max_score)
    over_max_score = False
    if offspring_population[0]['score'] > max_score:
      over_max_score = True
      max_score = offspring_population[0]['score']
      v_strong = deepcopy(offspring_population[0]['vector'])
    
    iteration_score_evolution.append(offspring_population[0]['score'])
    
    if over_max_score is True:
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score
    
    max_score_evolution.append(max_score)
    
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution)

  if convergence_score == 0: # if convergence not yet reached, convergence is considered to be max_score.
    convergence_iteration = max_score_iteration
    convergence_score = max_score
  return [
              convergence_iteration, 
              convergence_score, 
              max_score_iteration, 
              max_score, 
              res
          ]




# DEEP - 2015
# Works without a front of best solutions.
# Experimentations show that DEEP is super slow like all other front based algorithms.
# It struggles to reach best solutions too.
def deep():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  
  # Strongest vector
  v_strong = []
  # Score of the strongest vector
  max_score = 0
  
  iteration_score_evolution = []
  max_score_evolution = []

  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of max_score
  max_score_iteration = 0

  population = []
  center_former_population = 0
  center_new_population = 0

  # PARAMETERS
  f = 0.5   # f belong to [0, 2] and classic used value is 0.8.
  cr = 0.9  # cr belong to [0, 1] and classic used value is 0.9.
  population_size = 10
  s = 2     # number of best vectors we consider to calculate centerg.
  alpha_sigma = 0.3 # Standard deviation used to calculate alpha values.
  beta_sigma = 0.05 # Standard deviation used to calculate beta values.
  alpha_max = 2 # Max value of an alpha value.
  beta_max = 0.25 # Max value of a beta value.
  lambda_value = 0.5
  

  # INITIALIZATION
  # Generate the initial population.
  for i in range(0, population_size):
    # Create a vector.
    vector = generate_vector()
    
    vector_score = score(vector)
    # Add the vector to the population.
    population.append({
                        'vector':deepcopy(vector),
                        'score':vector_score,
                        'index':i
                      }) 
    # Add to top_vectors and all_vectors.
    add_to_top_vectors(deepcopy(vector), vector_score)   
    all_vectors.append(deepcopy(vector))
  
  # vep is the cumulative learning EP vector.
  # vep is defined as the cumulative migration step of the mean point of the population between successive generations.
  # vep is set as zero vector.
  vep = 0
  
  # cep is the anchor point.
  # cep is set as center0.
  # centerg is the mean position of the first s best individuals in the population in generation g. So center0 is the mean position of the first s best individuals in the population in generation 0.
  # Get center value :
  # Sort population by score DESC.
  population = sorted(deepcopy(population), key=lambda dct: dct['score'], reverse=True)
  sum_index_population = 0
  for k in range(0, s):
    sum_index_population += population[k]['index']
  center_population = sum_index_population / s
  cep = center_population
  center_former_population = center_population
  
  # alpham and betam are set as 0.
  alpham = 0
  betam = 0


  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations
    
    #print("Iteration N° " + str(i+1)) 
    
    # Creation of a mutated population
    mutated_population = []
    # Each individual in the mutated population goes through the DE mutation and crossover operations.
    for k in range(0, len(population)):
      # Mutation
      # The mutation operation of DE is very special in that it uses a linear combination of a base vector and one differential vector or more to generate a mutated vector.
      # Select three random vectors in population different from the current one treated.
      index_1 = k
      index_2 = k
      index_3 = k
      while index_1 == k:
        index_1 = random.randint(0, len(population)-1)
      while index_2 == k:
        index_2 = random.randint(0, len(population)-1)
      while index_3 == k:
        index_3 = random.randint(0, len(population)-1)
      random_vector_1 = population[index_1]['vector']
      random_vector_2 = population[index_2]['vector']
      random_vector_3 = population[index_3]['vector']
      # Difference between random_vector_2 and random_vector_3.
      diff_rv2_rv3 = [a_i - b_i for a_i, b_i in zip(deepcopy(random_vector_2), deepcopy(random_vector_3))]
      # diff_rv2_rv3 multiplied by f.
      diff_mult_f = [f*x for x in diff_rv2_rv3]
      # Mutated Vector
      mutated_vector = [a_i + b_i for a_i, b_i in zip(deepcopy(random_vector_1), diff_mult_f)]
      mutated_population.append({
                                  'vector':deepcopy(mutated_vector),
                                  'score':score(mutated_vector),
                                  'index':k
                                })
    
    # Crossover between population and mutated_population
    crossed_population = []
    for k in range(0, len(population)):
      crossed_vector = []
      for d in range(0, len(population[k]['vector'])):
        rand = random.uniform(0, 1) # random number uniformly distributed within [0, 1]
        d_rand = random.randint(0, len(population[k]['vector'])) # random index in the vector
        if rand <= cr or d == d_rand:
          crossed_vector.append(mutated_population[k]['vector'][d])
        else:
          crossed_vector.append(population[k]['vector'][d])
      crossed_population.append({
                                  'vector':deepcopy(crossed_vector),
                                  'score':score(crossed_vector),
                                  'index':k
                                })
    
    # Additional EP mutation of each vector in crossed_population.
    alpha_good = []
    beta_good = []
    for k in range(0, len(crossed_population)):
      alpha = 2*gauss(alpham, alpha_sigma)
      if alpha > alpha_max : alpha = alpha_max
      if alpha < -alpha_max : alpha = -alpha_max
      
      beta = gauss(betam, beta_sigma)
      if beta < 0 : beta = 0
      if beta > beta_max : beta = beta_max
      
      # Build new_vector.
      # cep - crossed_population[k]['vector']
      term_1 = [(-1)*x+cep for x in deepcopy(crossed_population[k]['vector'])]
      # beta * term_1
      term_2 = [beta*x for x in term_1]
      # alpha * vep
      term_3 = alpha * vep
      # term_3 + term_2
      term_4 = [term_3+x for x in term_2]
      # f * cr
      term_5 = f * cr
      # term_5 * term_4
      term_6 = [term_5*x for x in term_4]
      # crossed_population[k]['vector'] + term_6
      term_7 = [a_i + b_i for a_i, b_i in zip(deepcopy(crossed_population[k]['vector']), term_6)]
      new_vector = term_7
      """
      new_vector =      
        deepcopy(crossed_population[k]['vector']) 
        + 
        f*cr
        *
        (alpha*vep + beta*(cep-deepcopy(crossed_population[k]['vector'])))
      """
      
      # Evaluate the new individual.
      new_score = score(new_vector)
      # If new_score is better than the former score, then we save the alpha used in alpha_good.
      if new_score > crossed_population[k]['score']:
        alpha_good.append(alpha)
        beta_good.append(beta)
      # Update crossed_population[k].
      crossed_population[k]['vector'] = deepcopy(new_vector)
      crossed_population[k]['score'] = new_score
    
    
    # Normalise each vector in crossed_population.
    for k in range(len(crossed_population)):
      # Reshape crossed_population[k] so that it contains only 0-digits and 1-digits.
      for d in range(len(crossed_population[k]['vector'])):
        if crossed_population[k]['vector'][d] > 0:
          crossed_population[k]['vector'][d] = 1
        elif crossed_population[k]['vector'][d] <= 0:
          crossed_population[k]['vector'][d] = 0
      # Assure that crossed_population[k] respects constraints.
      crossed_population[k]['vector'] = normalise(deepcopy(crossed_population[k]['vector']))
      # Recalculate the score of crossed_population[k].
      crossed_population[k]['score'] = score(deepcopy(crossed_population[k]['vector']))    
    # Add vectors from crossed_population to top_vectors and all_vectors.
    for k in range(len(crossed_population)):
      add_to_top_vectors(deepcopy(crossed_population[k]['vector']), crossed_population[k]['score'])   
      all_vectors.append(deepcopy(crossed_population[k]['vector']))    
    
    
    # Do normal DE selection
    # Pair-wise fitness comparison of crossed_population and population.
    selected_population = []
    for k in range(0, len(population)):
      if population[k]['score'] > crossed_population[k]['score']:
        # Add population[k] into selected_population.
        selected_population.append(deepcopy(population[k]))
      else:        
        # Add crossed_population[k] into selected_population.
        selected_population.append(deepcopy(crossed_population[k]))
      selected_population[k]['index'] = k
    
    
    # UPDATES
    
    # Update alpham
    mean_alpha_good = 0
    if len(alpha_good) > 0: mean_alpha_good = mean(alpha_good)
    alpham = 0.9*alpham + 0.1*mean_alpha_good/2
    
    # Update betam
    mean_beta_good = 0
    if len(beta_good) > 0: mean_beta_good = mean(beta_good)
    betam = 0.9*betam + 0.1*mean_beta_good
    
    # Update vep
    # vep is the migration vector of the mean point of the population center_g from the current population (population) to the future population (selected_population).
    # Sort selected_population by score DESC.
    selected_population = sorted(deepcopy(selected_population), key=lambda dct: dct['score'], reverse=True)
    # Get centers : a center is the mean position of the first s best individuals in the population.
    sum_index_selected_population = 0
    for k in range(0, s):
      sum_index_selected_population += selected_population[k]['index']
    center_new_population = sum_index_selected_population / s
    vep = center_new_population - center_former_population
    
    # Update cep
    cep = lambda_value*cep + (1-lambda_value)*center_new_population

    # selected_population determined above is the population for next iteration.
    population = deepcopy(selected_population)
    center_former_population = center_new_population # for next iteration too
    
    
    # Strongest vector (v_strong) and associated score (max_score)
    over_max_score = False
    if selected_population[0]['score'] > max_score:
      over_max_score = True
      max_score = selected_population[0]['score']
      v_strong = deepcopy(selected_population[0]['vector'])
    
    iteration_score_evolution.append(selected_population[0]['score'])
    
    if over_max_score is True:
      max_score_iteration = i+1
      if convergence_score == 0: # if convergence not yet reached  
        count_stagnation = stagnation # reset counter
    else:
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score
    
    max_score_evolution.append(max_score)
    
  # Select and Display recommended resources for each actor.
  res = select_and_display_recommendations(iteration_score_evolution, max_score_evolution)

  if convergence_score == 0: # if convergence not yet reached, convergence is considered to be max_score.
    convergence_iteration = max_score_iteration
    convergence_score = max_score
  return [
              convergence_iteration, 
              convergence_score, 
              max_score_iteration, 
              max_score, 
              res
          ]




# SPEA2 - 2001
# NOTA : Regarding experimentations, this algorithm is super slow and struggles to reach maximized solutions, even with only three objectives.
def spea2():
  global top_vectors
  top_vectors = []
  global all_vectors
  all_vectors = []
  
  strongest_vector = []
  max_score = 0
  
  iteration_score_evolution = []
  max_score_evolution = []
  
  population = []
  population_size = 10
  
  # Besides the population, an archive is maintained which contains a representation of the nondominated front among all solutions considered so far. 
  # A member of the archive is only removed if :
  #   i) a solution has been found that dominates it.
  #   or 
  #   ii) the maximum archive size is exceeded and the portion of the front where the archive member is located is overcrowded. 
  # Usually, being copied to the archive is the only way how an vector can survive several generations in addition to pure reproduction which may occur by chance.
  # This technique is incorporated in order not to lose certain portions of the current non-dominated front due to random effects.
  archive = []
  archive_size = 5
  
  # For experimentations
  convergence_iteration = 0
  convergence_score = 0
  count_stagnation = stagnation # convergence if x iterations passed without an evolution of score_max 
  max_score_iteration = 0



  # Step 1: Initialization
  # Generate the initial population.
  for i in range(0, population_size):
    # Create a vector.
    vector = generate_vector()
    
    # Add the vector to the population.
    population.append(vector)

    # Adding to top_vectors and all_vectors.  
    vector_score = score(vector)
    add_to_top_vectors(vector, vector_score)    
    all_vectors.append(vector) 
  
  
  iteration_number = algorithm_iterations
  for i in range(iteration_number): # x iterations
    
    # print("Iteration N° " + str(i+1))

    # Step 2: Fitness assignment.

    # Calculate fitness values of vectors in population and archive.
    # More detailed explanation :
      # To each vector in the archive and the population is assigned a strength_value representing the number of solutions it dominates.
      # Here is how to spot if a vector is dominated by another one :
      # To each vector is associated a score array like [0.2, 0.3] meaning here that there are 2 scores/objectives (2 values in the array). A vector is dominated by another one if at least one of its scores is inferior to the corresponding one in the other vector and if all of its scores are inferior or equal to the corresponding ones in the other vector.
    """  
      # For example, let's consider :
      X1 = [0, 0.5]
      X2 = [0.5, 0.5]
      
      # Return True if X1 is dominated by X2.
      def dominates(X1, X2):
      if(np.any(X1 < X2) and np.all(X1 <= X2)):
          return True
      else:
          return False
    """  
    # In our case, we have summed all the scores into one so we just have to compare scores between vectors to spot dominance.
    
    # Do it for the population.
    population_strength_values = []
    population_dominated_indicators = []
    for v_ref in population:
      strength_value = 0
      scores_v_ref = [score(v_ref)]
      population_dominated_indicators.append(False)
      for v_comp in population:
        scores_v_comp = [score(v_comp)]
        if(numpy_any(scores_v_comp < scores_v_ref) and numpy_all(scores_v_comp <= scores_v_ref)):
          # v_ref dominates v_comp so we increment its strength_value.
          strength_value += 1
        elif(numpy_any(scores_v_ref < scores_v_comp) and numpy_all(scores_v_ref <= scores_v_comp)): 
          # v_comp dominates v_ref so we mark v_ref as "dominated".
          index = len(population_dominated_indicators) - 1
          population_dominated_indicators[index] = True
      population_strength_values.append(strength_value)
        
    # Do it for the archive.
    archive_strength_values = []
    archive_dominated_indicators = []
    for v_ref in archive:
      strength_value = 0
      scores_v_ref = [score(v_ref)]
      archive_dominated_indicators.append(False)
      for v_comp in archive:
        scores_v_comp = [score(v_comp)]
        if(numpy_any(scores_v_comp < scores_v_ref) and numpy_all(scores_v_comp <= scores_v_ref)):
          # v_ref dominates v_comp so we increment its strength_value.
          strength_value += 1
        elif(numpy_any(scores_v_ref < scores_v_comp) and numpy_all(scores_v_ref <= scores_v_comp)): 
          # v_comp dominates v_ref so we mark v_ref as "dominated".
          index = len(archive_dominated_indicators) - 1
          archive_dominated_indicators[index] = True
      archive_strength_values.append(strength_value)     


    # Step 3: Environmental selection.
    
    # Copy all non-dominated vectors in population and archive to new_archive. 
    new_archive = []
    index = 0
    for bool_dominated in population_dominated_indicators:
      if bool_dominated == False:
        new_archive.append(population[index])
      index += 1
    
    index = 0
    for bool_dominated in archive_dominated_indicators:
      if bool_dominated == False:
        new_archive.append(archive[index])
      index += 1
    
    # If size of new_archive exceeds archive_size then iteratively remove vectors from new_archive until size of new_archive is equal to archive_size.
    # For each step :
      # Take the distance of each vector to its k-th nearest neighbor or to the vector immediately superior to him.
      # The vector which has the minimum distance to another vector is removed.
      # If there are several vectors with minimum distance, we consider the second smallest distances and so forth.
    # Let's go for it.
    # Sort new_archive by score ASC (the key is the score() function).
    new_archive = sorted(deepcopy(new_archive), key=score)
    while len(new_archive) > archive_size:
      # Get distances.
      distances = []
      index = 0
      for v in new_archive:
        if index < len(new_archive)-1:
          distance = score(new_archive[index+1]) - score(v)
          distances.append(distance)
        else:
          distances.append(1000) # For the last element having the highest score.
        index += 1
      # Associate distances to vectors.
      vectors_and_distances = []
      index = 0
      for v in new_archive:
        vectors_and_distances.append({
                                        'vector_index_in_new_archive':index,
                                        'distance':distances[index]
                                    }) 
        index += 1
      # Sort vectors_and_distances by distance ASC.
      vectors_and_distances = sorted(deepcopy(vectors_and_distances), key=lambda dct: dct['distance'])
      # Remove the vector having the minimum distance.
      index_to_remove = vectors_and_distances[0]['vector_index_in_new_archive']
      new_archive.pop(index_to_remove)
    
    
    # If size of new_archive is less than archive_size then the best dominated vectors in the previous archive and population are copied to new_archive until new_archive is filled.
    # Let's go for it.
    # Merge dominated vectors from population and archive into dominated_ones.
    dominated_ones = []
    index = 0
    for v in population:
      if population_dominated_indicators[index] == True:
        dominated_ones.append(v)
      index += 1
    index = 0
    for v in archive:
      if archive_dominated_indicators[index] == True:
        dominated_ones.append(v)
      index += 1
    # Sort dominated_ones by score DESC (the key is the score() function).
    dominated_ones = sorted(deepcopy(dominated_ones), key=score, reverse=True)
    # Fill new_archive until its size is equal to archive_size.
    while len(new_archive) < archive_size:
      new_archive.append(dominated_ones[0])
      dominated_ones.pop(0)


    # Step 4: Termination.
    # Set non_dominated_ones = the set of non-dominated vectors in new_archive
    
    # Spot dominance in new_archive.
    new_archive_strength_values = []
    new_archive_dominated_indicators = []
    for v_ref in new_archive:
      strength_value = 0
      scores_v_ref = [score(v_ref)]
      new_archive_dominated_indicators.append(False)
      for v_comp in new_archive:
        scores_v_comp = [score(v_comp)]
        if(numpy_any(scores_v_comp < scores_v_ref) and numpy_all(scores_v_comp <= scores_v_ref)):
          # v_ref dominates v_comp so we increment its strength_value.
          strength_value += 1
        elif(numpy_any(scores_v_ref < scores_v_comp) and numpy_all(scores_v_ref <= scores_v_comp)): 
          # v_comp dominates v_ref so we mark v_ref as "dominated".
          index = len(new_archive_dominated_indicators) - 1
          new_archive_dominated_indicators[index] = True
      new_archive_strength_values.append(strength_value)

    # Feed non_dominated_ones with non-dominated vectors from new_archive.
    non_dominated_ones = []
    index = 0
    for bool_dominated in new_archive_dominated_indicators:
      if bool_dominated == False:
        non_dominated_ones.append(new_archive[index])
      index += 1

    
    # Step 5: Mating selection.
    # Perform binary tournament selection on new_archive in order to fill the mating pool.
    # More details :
      # Do the following a number of times equal to the population size.
        # Randomly pick two vectors from new_archive.
        # Put the one having the best score into the mating pool.
    mating_pool = []
    for k in range(0, population_size):
      index_1 = random.randint(0, len(new_archive)-1)
      index_2 = random.randint(0, len(new_archive)-1)
      if score(new_archive[index_1]) >= score(new_archive[index_2]):
        mating_pool.append(new_archive[index_1])
      else:
        mating_pool.append(new_archive[index_2])
    

    # Step 6: Crossovers + Mutations on the mating pool.
    
    # Crossover type : on two vectors (parent) with one-point crossover. This means that we select a random index, take the right part of the first parent before index and the left part of the second parent after index to create the first child. We do it vice-versa to get the second child.
    for k in range(0, int(population_size/2)):
      parent_1 = mating_pool[0]
      parent_2 = mating_pool[1]
      mating_pool.pop(0)
      mating_pool.pop(1)
      index = random.randint(0, len(parent_1)-1)
      child_1 = parent_1
      for y in range(index, len(child_1)-1): 
        child_1[y] = parent_2[y]
      child_2 = parent_2
      for y in range(index, len(child_2)-1): 
        child_2[y] = parent_1[y]
      mating_pool.append(child_1)
      mating_pool.append(child_2)
    # The mating_pool is now full of children.
    
    # Mutation type : in each vector of the mating pool, each digit is flipped with a probability of 0.006.
    for v in range(0, len(mating_pool)):
      for d in range(0, len(mating_pool[v])):
        rand = random.randint(1, 1000)
        if rand <= 6:
          if mating_pool[v][d] == 0: 
            mating_pool[v][d] = 1
          else: 
            mating_pool[v][d] = 0

    # Assure that each vector of mating pool respect constraints.
    for v in range(len(mating_pool)):
      mating_pool[v] = normalise(deepcopy(mating_pool[v]))
      
    # Assure that each vector of new_archive respect constraints.
    for v in range(len(new_archive)):
      new_archive[v] = normalise(deepcopy(new_archive[v]))
         
    # Add in top_vectors and all_vectors.
    # Add all vectors from mating_pool.
    for k in range(len(mating_pool)): 
      vector_score = score(mating_pool[k])
      add_to_top_vectors(mating_pool[k], vector_score)   
      all_vectors.append(mating_pool[k])
    # Add all vectors from new_archive.
    for k in range(len(new_archive)): 
      vector_score = score(new_archive[k])
      add_to_top_vectors(new_archive[k], vector_score)   
      all_vectors.append(new_archive[k])      
    
    # Population and archive for next iteration.
    # The final mating pool obtained is the population for next iteration. 
    population = deepcopy(mating_pool)
    # new_archive is the archive for next iteration.
    archive = deepcopy(new_archive)

    # Concatenate new_archive and mating_pool.
    # => for EVALUATIONS - ITERATION END
    final_pool = new_archive + mating_pool

    # EVALUATIONS - ITERATION END       
    powers_vectors = []
    over_max_score = False
    for k in range(len(final_pool)):
      power_k = score(final_pool[k])
      powers_vectors.append(power_k)  
      if power_k > max_score:
        over_max_score = True
        max_score = power_k
        strongest_vector = deepcopy(final_pool[k])
        max_score_iteration = i+1
        if convergence_score == 0: # if convergence not yet reached  
          count_stagnation = stagnation # reset counter
          
    if over_max_score is False:
      count_stagnation -= 1
      if count_stagnation == 0 and convergence_score == 0: # 1st convergence reached
        convergence_iteration = max_score_iteration
        convergence_score = max_score
    
    max_score_evolution.append(max_score)
    
    iteration_score_evolution.append(
                                      max(powers_vectors)
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
            res
          ]
        
        
        




















