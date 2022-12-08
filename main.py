# Imports
from past.builtins import execfile
execfile('imports.py')
execfile('reshape.py')
execfile('functions.py')
#execfile('state_of_art_algorithms.py')
# exec(open("./filename").read()) # => for Unix machines



# Get parameter df.
df = []
with open(sys.argv[1], newline='') as csvfile:
  df = list(csv.reader(csvfile))
# Transform df columns into floats.
for row in range(0, len(df)):
  for column in range(0, len(df[row])):
    df[row][column] = float(df[row][column])


# Criteria Weighting by creating an array containing the weights.
# A weight represents the mean percentage of participation of a criterion in the score of a resource.
# Weights are used inside the fitness functions (score() and resource_score()).
weights = []
for column_index in range(1, len(df[0])):
  weight_criterion = 0
  for k in range(0, len(df)):
    to_add = df[k][column_index]/sum(df[k][-(len(df[0])-1):])
    # Check if to_add is nan, corresponding to data rows where all attributes values are 0. In those cases we consider that the weight to add is 1.
    if math.isnan(to_add):
      weight_criterion += 1
    else:
      weight_criterion += to_add
  weight_criterion = weight_criterion/len(df)
  weights.append(weight_criterion)

 
# Get parameter df_indexes_and_popularity.
df_indexes_and_popularity = []
with open(sys.argv[9], newline='') as csvfile:
  df_indexes_and_popularity = list(csv.reader(csvfile))
# Transform columns into integers.
for row in range(0, len(df_indexes_and_popularity)):
  df_indexes_and_popularity[row][0] = int(df_indexes_and_popularity[row][0])
  df_indexes_and_popularity[row][1] = int(1000*float(df_indexes_and_popularity[row][1]))  
 
# Get parameter popular_items_indexes.
raw_popular_items_indexes = []
popular_items_indexes = []
with open(sys.argv[11], newline='') as csvfile:
  raw_popular_items_indexes = list(csv.reader(csvfile))
# Transform into integers.
for row in range(0, len(raw_popular_items_indexes)):
  popular_items_indexes.append(int(raw_popular_items_indexes[row][0]))


# Get parameter constraints.
constraints = []
with open(sys.argv[12], newline='') as csvfile:
  constraints = list(csv.reader(csvfile))
# Transform columns into integers or floats.
for row in range(len(constraints)):
  constraints[row][0] = int(constraints[row][0])    # index_in_df must be an integer.
  constraints[row][1] = float(constraints[row][1])  # max_value must be a float.
  constraints[row][2] = float(constraints[row][2])  # min_value must be a float.





# Get other parameters/Global_Variables.
# Reminder : In Python, accessing global variables inside functions is ok but if we want to MODIFY a global variable inside a function, we must declare it as global inside the function or use a singleton pattern.

# Macro iterations.
macro_iterations = int(sys.argv[2])

# Algorithm iterations.
algorithm_iterations = int(sys.argv[3])

# Iterations needed without an evolution of max score to say that convergence is reached.
stagnation = int(sys.argv[4])

# This is the number of actors concerned by the recommendation. Each actor get a vector as recommendation.
actors_number = int(sys.argv[5])

# This is the minimum value of similarity between recommended vectors of different actors.
similarity_percentage = float(sys.argv[6]) 

# Array containing all vectors generated through an algorithm iterations.
all_vectors = []

# Array of vectors kept for final recommendations. This array has its own feeding process, independant from each algorithm main process (see function add_to_top_vectors()).
top_vectors = [] 

# Used for similarity. This is the max number of vectors in top_vectors used to create the best combination of vectors regarding similarity.
max_top_vectors = int(sys.argv[7])

# Algorithms to execute. Also used for files saved.
# Available algorithms : "Random", "Genetic", "Ramify", "Struggle", "Spycle", "Spasm", "Extossom", "U3S", "EEP", "SPEA2", "DEEP", "DE", "EDA", "DE_EDA", "DCEA"
# If an algorithm name has a suffix "_ee_guided", it will be executed with guided exploration and exploitation. => Do not support all algorithms.
# Additional suffixes can be added to the genetic algorithm to use different guiding functions.
algo_names = (sys.argv[8]).split("-")

# Percentage of best popularities of df_indexes_and_popularity to be considered as the head. The rest will be the tail.
popularity_head_percentage = float(sys.argv[10])

# Boolean to know if we want to guide exploration and exploitation while executing certain algorithms.
ee_guided = False

# Best and worse digits in df regarding score and constraints score.
# Used by the function which guides E&E with the potential.
best_worse_digits = get_best_and_worse_digits(4)

# Constraints completion scores of df digits.
# Used as parameter for the function which guides E&E with the constraints completion.
digits_cc_scores = []

# ------------------------------
# E&E Power Global Variables
# ------------------------------
# Array of discovered digits ("dd").
# Array of non-discovered digits ("ndd").
# These arrays are used to keep track of when digits are discovered throughout an algorithm iterations. If a digit is discovered, it leaves "ndd" to go into "dd". 
# These arrays must be reseted before each new macro-iteration.
dd = []
ndd = []
# ------------------------------
# Counters of iterations that allowed to discover new digits (exploration) and that didn't allow to discover new digits (exploitation).
# These counters must be reseted before each new macro-iteration.
counter_explore_iter = 0
counter_exploit_iter = 0
# ------------------------------
# Counters of vectors that allowed to discover new digits (exploration) and that didn't allow to discover new digits (exploitation).
# These counters must be reseted before each new macro-iteration.
counter_explore_vector = 0
counter_exploit_vector = 0
# ------------------------------
# Counters of newly discovered digits and of already discovered digits.
# These counters must be reseted before each new macro-iteration.
counter_explore_digit = 0
counter_exploit_digit = 0
# ------------------------------

# Exploration Iterations Timers.
# These timers are used for the temporal representation of E&E.
# This array must be reseted before each new macro-iteration.
explore_iter_timers = []

# Number of times df has been reshaped.
reshape_counter = int(sys.argv[14])

# Number of resources in df reshaped.
df_final_size = int(sys.argv[15])

# Profile Number
profile_number = int(sys.argv[13])

# Currently executed algorithm
current_algo = ""

# Reshape Function Used
reshape_function = sys.argv[16]

# Constraints_variation_name
constraints_variation_name = sys.argv[17]

# E&E Logs
ee_logs = []

# Array of classes fed with vectors. If a vector leads to the creation of a new class, we consider that it is an exploration vector and that the iteration where it was created is an exploration iteration. Other vectors that are just added to existing classes, are considered as exploitation vectors and associated iterations are considered as exploitation iterations.
vectors_classes = []
ee_phases = []
for k in range(algorithm_iterations):
  ee_phases.append("")

# Mean values of df resources.
df_mean_values = []
for k in range(len(df)):
  mean_value = 0
  for c in range(1, len(df[k])):
    mean_value += df[k][c]
  mean_value /= (len(df[0])-1) # Number of columns except the first one.
  df_mean_values.append(mean_value)
    
# HVV_max - Maximum Hyper-Volume considering dimensions max values.
# This is the maximum volume of the search space, calculated by multiplying df columns max values.
dimensions_values_max = []
df_matrix = np_matrix(df)
maxes = array(df_matrix.max(0))[0] # Get max value of each df column.
hvv_max = 1
for k in range(1, len(maxes)):
  qty = maxes[k]
  hvv_max *= qty
  # Add to dimensions_values_max.
  dimensions_values_max.append(qty)
  
# HVO_max - Maximum Hyper-Volume considering dimensions occurrences.
# This is the maximum volume of the search space, calculated by multiplying df columns number of distinct values.
dimensions_occurrences_max = []
df_transposed = df_matrix.T
hvo_max = 1
for k in range(1, len(df_transposed)):
  # Get the unique values of the dimension considered by converting the list to a set, calculate its len and multiply hvo_max by it.
  qty = len(set(df_transposed[k].tolist()[0]))
  hvo_max *= qty
  # Add to dimensions_occurrences_max.
  dimensions_occurrences_max.append(qty)




# EXCEL FILE
# Workbook() takes one, non-optional, argument  
# which is the filename that we want to create. 
workbook = xlsxwriter.Workbook('Experiments/RESULTS_profile'+str(profile_number)+'_'+reshape_function+'.xlsx') 
# The workbook object is then used to add new  
# worksheet via the add_worksheet() method. 
worksheet = workbook.add_worksheet()
worksheet.write('A1', "Macro Loops = "+str(macro_iterations)+" Algo Loops = "+str(algorithm_iterations)+" Stagnation = "+str(stagnation)) 
worksheet.write('B1', "reshapes done") 
worksheet.write('C1', "aftermath digits") 
worksheet.write('D1', "ci") 
worksheet.write('E1', "msi") 
worksheet.write('F1', "cs")
worksheet.write('G1', "ms")
worksheet.write('H1', "et")
#worksheet.write('G1', "real_mean_sim")
#worksheet.write('G1', "real_mean_score")
worksheet.write('I1', "presence rate")
worksheet.write('J1', "all_vectors_coverage")
worksheet.write('K1', "top_vectors_coverage")
worksheet.write('L1', "popularity")
worksheet.write('M1', "constraints_completion")
worksheet.write('N1', "potential")
worksheet.write('O1', "links gap")
worksheet.write('P1', "vector links gap")

worksheet.write('Q1', "ee_power")
#worksheet.write('P1', "explore iter")
#worksheet.write('Q1', "exploit iter")
#worksheet.write('R1', "explore v")
#worksheet.write('S1', "exploit v")
#worksheet.write('T1', "explore d")
#worksheet.write('U1', "exploit d")
worksheet.write('R1', "min_eit_gap")
worksheet.write('S1', "max_eit_gap")
worksheet.write('T1', "diff_min_max_eit_gaps")
worksheet.write('U1', "mean_eit_gap")
worksheet.write('V1', "mean_diff_between_eit_gaps")

row_spacer = 2


# GO FOR EXECUTIONS    
for algo_name in algo_names:
  current_algo = algo_name
  ee_guided = False
  results = []
  for r in range(macro_iterations):
    
    print("")
    print(algo_name + " - P" + str(profile_number) + " - " + reshape_function.upper() + " - MACRO LOOP " + str(r+1))
    
    # Get time in seconds.
    time_start = time.perf_counter()
    
    # Reset "dd", "ndd" and E&E counters.
    reset_dd_ndd_counters()
    
    # Reset explore_iter_timers.
    explore_iter_timers = []
    
    # Reset ee_logs.
    ee_logs = []
    
    # Reset vectors_classes and ee_phases
    vectors_classes = []
    ee_phases = []
    for k in range(algorithm_iterations):
      ee_phases.append("")
    
    # Execute algorithm.
    
    if algo_name == "Combi_reduc":
      result = combi_reduc()
    
    elif algo_name == "Reinforcement":
      result = reinforcement()
      
    elif algo_name == "Reinforcement_reshape":
      result = reinforcement()
      
    elif algo_name == "Reinforcement_ee_guided_separated":
      ee_guided = True
      guide_function = "separated_EE_cases"
      result = reinforcement()
    elif algo_name == "Reinforcement_ee_guided_merged":
      ee_guided = True
      guide_function = "merged_EE_cases"
      result = reinforcement()
    elif algo_name == "Reinforcement_ee_guided_potential":
      ee_guided = True
      guide_function = "potential"
      result = reinforcement()
    elif algo_name == "Reinforcement_ee_guided_cc":
      ee_guided = True
      guide_function = "cc"
      digits_cc_scores = get_digits_cc_scores()
      result = reinforcement()
    elif algo_name == "Reinforcement_ee_guided_links":
      ee_guided = True
      guide_function = "links"
      result = reinforcement()
    elif algo_name == "Reinforcement_ee_guided_vector_links":
      ee_guided = True
      guide_function = "vector_links"
      result = reinforcement()
    elif algo_name == "Reinforcement_ee_guided_power":
      ee_guided = True
      guide_function = "power"
      result = reinforcement()  
    elif algo_name == "Reinforcement_ee_guided_temporal_balance":
      ee_guided = True
      guide_function = "temporal_balance"
      result = reinforcement() 
   
    elif algo_name == "Random":
      result = random_()
      
    elif algo_name == "Genetic":
      result = genetic()
      
    elif algo_name == "Genetic_ee_logged":
      result = genetic_ee_logged()
    elif algo_name == "Reinforcement_ee_logged":
      result = reinforcement_ee_logged()
    elif algo_name == "Ramify_ee_logged":
      result = ramify_ee_logged() 
    elif algo_name == "Struggle_ee_logged":
      result = struggle_ee_logged() 
    elif algo_name == "Spycle_ee_logged":
      result = spycle_ee_logged() 
    elif algo_name == "Spasm_ee_logged":
      result = spasm_ee_logged() 
    elif algo_name == "Extossom_ee_logged":
      result = extossom_ee_logged() 
    elif algo_name == "U3S_ee_logged":
      result = u3s_ee_logged()       
    elif algo_name == "EEP_ee_logged":
      result = eep_ee_logged()      
      
    elif algo_name == "Genetic_ee_guided_separated":
      ee_guided = True
      guide_function = "separated_EE_cases"
      result = genetic()
    elif algo_name == "Genetic_ee_guided_merged":
      ee_guided = True
      guide_function = "merged_EE_cases"
      result = genetic()
    elif algo_name == "Genetic_ee_guided_potential":
      ee_guided = True
      guide_function = "potential"
      result = genetic()
    elif algo_name == "Genetic_ee_guided_cc":
      ee_guided = True
      guide_function = "cc"
      digits_cc_scores = get_digits_cc_scores()
      result = genetic()
    elif algo_name == "Genetic_ee_guided_links":
      ee_guided = True
      guide_function = "links"
      result = genetic()
    elif algo_name == "Genetic_ee_guided_vector_links":
      ee_guided = True
      guide_function = "vector_links"
      result = genetic()
    elif algo_name == "Genetic_ee_guided_power":
      ee_guided = True
      guide_function = "power"
      result = genetic()  
    elif algo_name == "Genetic_ee_guided_temporal_balance":
      ee_guided = True
      guide_function = "temporal_balance"
      result = genetic() 
      
    elif algo_name == "Genetic_reshape":
      result = genetic()
    elif algo_name == "Genetic_dynashape":
      result = genetic_dynashape()
     
    
    elif algo_name == "Ramify":
      result = ramify()
    elif algo_name == "Ramify_ee_guided":
      ee_guided = True
      result = ramify()
    elif algo_name == "Struggle":
      result = struggle()
    elif algo_name == "Struggle_ee_guided":
      ee_guided = True
      result = struggle()
    elif algo_name == "Spycle":
      result = spycle()
    elif algo_name == "Spycle_ee_guided":
      ee_guided = True
      result = spycle()
    elif algo_name == "Spasm":
      result = spasm()
    elif algo_name == "Spasm_ee_guided":
      ee_guided = True
      result = spasm()
    elif algo_name == "Extossom":
      result = extossom()
    elif algo_name == "Extossom_ee_guided":
      ee_guided = True
      result = extossom()
    elif algo_name == "U3S":
      result = u3s()
    elif algo_name == "EEP":
      result = eep()
    
    elif algo_name == "SPEA2":
      result = spea2()
    elif algo_name == "DEEP":
      result = deep()
    elif algo_name == "DE":
      result = de()
    elif algo_name == "EDA":
      result = eda()
    elif algo_name == "DE_EDA":
      result = de_eda()
    elif algo_name == "DCEA":
      result = dcea()


    # Get time in seconds
    time_end = time.perf_counter()
    # Save the execution time of one macro loop.
    exec_time = time_end - time_start
    result.append(exec_time)
    #print("Execution Time : " + str(exec_time))
    
    results.append(result)
    
    # Check if all vectors in top_vectors respect constraints.
    for k in range(len(top_vectors)):
      if constraints_check(top_vectors[k]['vector']) is False:
        #print(top_vectors[k]['vector'])
        print("ERROR - A vector in top_vectors does not respect constraints.")
        sys.exit()

  """
  # --------------------
  # DEBUG
  # --------------------
  #for k in range(len(vectors_classes)):
  #  ic(len(vectors_classes[k]))
  ic(len(vectors_classes))
  count_exploration = 0
  for k in range(len(ee_phases)):
    if ee_phases[k] == "Exploration":
      count_exploration += 1
  print(str(count_exploration) + " Exploration Phases / " + str(len(ee_phases)) + " Phases")
  #print(ee_phases)
  # --------------------
  """



  # STATISTICAL TESTS - SAVES AND PREPARATION
  # If the algorithm is "Genetic" or "Reinforcement", save its results object in a file in order to use them as the baseline for statistical tests. The word "baseline" is used below to refer to "Genetic" or "Reinforcement".
  id = sys.argv[13]
  baseline_results = None
  if algo_name == "Genetic" or algo_name == "Reinforcement":
    file = open('Experiments/baseline_results_id_' + str(id) + '.obj', 'wb') 
    pickle.dump(results, file)
  # Else, get these results from file. They must obviously have been saved formerly for the baseline algorithm and for this id. These results will be used for statistical tests.
  else:
    file = open('Experiments/baseline_results_id_' + str(id) + '.obj', 'rb') 
    baseline_results = pickle.load(file)
    
  # Get all baseline results ready to be passed to the statistical test function.
  b_convergence_iterations = []
  b_convergence_scores = []
  b_max_score_iterations = []
  b_max_scores = []
  #b_real_mean_similarities = []
  #b_real_mean_scores = []
  b_dense_vectors = []
  b_presence_gaps = []
  b_presence_rates = []
  b_all_vectors_coverages = []
  b_top_vectors_coverages = []
  b_popularities = []
  b_constraints_completions = []
  b_potentials = []
  b_links_gaps = []
  b_vector_links_gaps = []
  b_execution_times = []
  
  b_ee_powers = []
  #b_counters_explore_iter = []
  #b_counters_exploit_iter = []
  #b_counters_explore_vector = []
  #b_counters_exploit_vector = []
  #b_counters_explore_digit = []
  #b_counters_exploit_digit = []
  
  b_min_eit_gap_s = []
  b_max_eit_gap_s = []
  b_diff_min_max_eit_gaps_s = []
  b_mean_eit_gap_s = []
  b_mean_diff_between_eit_gaps_s = []
  
  if algo_name != "Genetic" and algo_name != "Reinforcement":
    for r in baseline_results: 
      b_convergence_iterations.append(r[0])
      b_convergence_scores.append(r[1])  
      b_max_score_iterations.append(r[2]) 
      b_max_scores.append(r[3]) 
      #b_real_mean_similarities.append(r[4][0])
      #b_real_mean_scores.append(r[4][0])
      b_dense_vectors.append(r[4][0])
      b_presence_gaps.append(r[4][1])
      b_presence_rates.append(r[4][2])
      b_all_vectors_coverages.append(r[4][3])
      b_top_vectors_coverages.append(r[4][4])
      b_popularities.append(r[4][5])
      b_constraints_completions.append(r[4][6])
      b_potentials.append(r[4][7])
      b_links_gaps.append(r[4][8])
      b_vector_links_gaps.append(r[4][9])
      b_execution_times.append(r[5])
    
      b_ee_powers.append(r[4][10])
      #b_counters_explore_iter.append(r[4][11])
      #b_counters_exploit_iter.append(r[4][12])
      #b_counters_explore_vector.append(r[4][13])
      #b_counters_exploit_vector.append(r[4][14])
      #b_counters_explore_digit.append(r[4][15])
      #b_counters_exploit_digit.append(r[4][16])

      b_min_eit_gap_s.append(r[4][11])
      b_max_eit_gap_s.append(r[4][12])
      b_diff_min_max_eit_gaps_s.append(r[4][13])
      b_mean_eit_gap_s.append(r[4][14])
      b_mean_diff_between_eit_gaps_s.append(r[4][15])


  # Get each element separately from results.
  convergence_iterations = []
  convergence_scores = []
  max_score_iterations = []
  max_scores = []
  #real_mean_similarities = []
  #real_mean_scores = []
  dense_vectors = []
  presence_gaps = []
  presence_rates = []
  all_vectors_coverages = []
  top_vectors_coverages = []
  popularities = []
  constraints_completions = []
  potentials = []
  links_gaps = []
  vector_links_gaps = []
  execution_times = []
  
  ee_powers = []
  #counters_explore_iter = []
  #counters_exploit_iter = []
  #counters_explore_vector = []
  #counters_exploit_vector = []
  #counters_explore_digit = []
  #counters_exploit_digit = []
  
  min_eit_gap_s = []
  max_eit_gap_s = []
  diff_min_max_eit_gaps_s = []
  mean_eit_gap_s = []
  mean_diff_between_eit_gaps_s = []
  
  for r in results: 
    convergence_iterations.append(r[0])
    convergence_scores.append(r[1])  
    max_score_iterations.append(r[2]) 
    max_scores.append(r[3]) 
    #real_mean_similarities.append(r[4][0])
    #real_mean_scores.append(r[4][0])
    dense_vectors.append(r[4][0])
    presence_gaps.append(r[4][1])
    presence_rates.append(r[4][2])
    all_vectors_coverages.append(r[4][3])
    top_vectors_coverages.append(r[4][4])
    popularities.append(r[4][5])
    constraints_completions.append(r[4][6])
    potentials.append(r[4][7])
    links_gaps.append(r[4][8])
    vector_links_gaps.append(r[4][9])
    execution_times.append(r[5])
    
    ee_powers.append(r[4][10])
    #counters_explore_iter.append(r[4][11])
    #counters_exploit_iter.append(r[4][12])
    #counters_explore_vector.append(r[4][13])
    #counters_exploit_vector.append(r[4][14])
    #counters_explore_digit.append(r[4][15])
    #counters_exploit_digit.append(r[4][16])
    
    min_eit_gap_s.append(r[4][11])
    max_eit_gap_s.append(r[4][12])
    diff_min_max_eit_gaps_s.append(r[4][13])
    mean_eit_gap_s.append(r[4][14])
    mean_diff_between_eit_gaps_s.append(r[4][15])
    
  
  # Mean Values
  mean_msi = round(mean(max_score_iterations))
  mean_ci = round(mean(convergence_iterations))
  mean_ms = round(mean(max_scores), 1)
  mean_cs = round(mean(convergence_scores), 1)
  mean_et = round(mean(execution_times), 1)
  #mean_rmsim = round(int(100*mean(real_mean_similarities)))
  #mean_rmscores = round(mean(real_mean_scores), 1)
  mean_dense_vector = np_mean(dense_vectors, axis = 0)
  mean_presence_gap = np_mean(presence_gaps, axis = 0)
  mean_presence_rate = int(100*mean(presence_rates))
  mean_avcov = round(int(100*mean(all_vectors_coverages)))
  mean_tvcov = round(int(100*mean(top_vectors_coverages)))
  mean_pop = round(int(100*mean(popularities)))
  mean_const_comp = round(int(100*mean(constraints_completions)))
  mean_potential = round(int(100*mean(potentials)))
  mean_links_gaps = round(int(100*mean(links_gaps)))
  mean_vector_links_gaps = round(int(100*mean(vector_links_gaps)))
  
  mean_ee_powers = int(100*mean(ee_powers))
  #mean_counters_explore_iter = int(mean(counters_explore_iter))
  #mean_counters_exploit_iter = int(mean(counters_exploit_iter))
  #mean_counters_explore_vector = int(mean(counters_explore_vector))
  #mean_counters_exploit_vector = int(mean(counters_exploit_vector))
  #mean_counters_explore_digit = int(mean(counters_explore_digit))
  #mean_counters_exploit_digit = int(mean(counters_exploit_digit))
  
  mean_of_min_eit_gap_s = round(mean(min_eit_gap_s), 2)
  mean_of_max_eit_gap_s = round(mean(max_eit_gap_s), 2)
  mean_of_diff_min_max_eit_gaps_s = round(mean(diff_min_max_eit_gaps_s), 2)
  mean_of_mean_eit_gap_s = round(mean(mean_eit_gap_s), 2)
  mean_of_mean_diff_between_eit_gaps_s = round(mean(mean_diff_between_eit_gaps_s), 2)
  
  """
  # Entropy Values
  entropy_msi = "Low"
  if max(max_score_iterations) - min(max_score_iterations) > 300:
    entropy_msi = "High"
  entropy_ci = "Low"
  if max(convergence_iterations) - min(convergence_iterations) > 300:
    entropy_ci = "High"
  entropy_ms = "Low"
  if max(max_scores) - min(max_scores) > 2:
    entropy_ms = "High"
  entropy_cs = "Low"
  if max(convergence_scores) - min(convergence_scores) > 2:
    entropy_cs = "High"
  entropy_et = "Low"
  if max(execution_times) - min(execution_times) > 10:
    entropy_et = "High"
  
  #entropy_rmsim = "Low"
  #if max(real_mean_similarities) - min(real_mean_similarities) > 0.3:
    #entropy_rmsim = "High"
  #entropy_rmscores = "Low"
  #if max(real_mean_scores) - min(real_mean_scores) > 2:
    #entropy_rmscores = "High"
  
  entropy_presence_rate = "Low"
  if max(presence_rates) - min(presence_rates) > 0.3:
    entropy_presence_rate = "High"
  entropy_avcov = "Low"
  if max(all_vectors_coverages) - min(all_vectors_coverages) > 0.3:
    entropy_avcov = "High"
  entropy_tvcov = "Low"
  if max(top_vectors_coverages) - min(top_vectors_coverages) > 0.3:
    entropy_tvcov = "High" 
  entropy_pop = "Low"
  if max(popularities) - min(popularities) > 0.3:
    entropy_pop = "High"
  entropy_const_comp = "Low"
  if max(constraints_completions) - min(constraints_completions) > 0.3:
    entropy_const_comp = "High"
  entropy_potential = "Low"
  if max(potentials) - min(potentials) > 0.3:
    entropy_potential = "High"
  entropy_links_gaps = "Low"
  if max(links_gaps) - min(links_gaps) > 0.3:
    entropy_links_gaps = "High"
  entropy_vector_links_gaps = "Low"
  if max(vector_links_gaps) - min(vector_links_gaps) > 0.3:
    entropy_vector_links_gaps = "High"
  """


  # ALL DATA + Charts Titles
  # Used for statistical tests, see below.
  baseline_data = [b_convergence_iterations, b_max_score_iterations, b_convergence_scores, b_max_scores, b_execution_times, b_presence_rates, b_all_vectors_coverages, b_top_vectors_coverages, b_popularities, b_constraints_completions, b_potentials, b_links_gaps, b_vector_links_gaps, b_ee_powers,
  b_min_eit_gap_s,
  b_max_eit_gap_s,
  b_diff_min_max_eit_gaps_s,
  b_mean_eit_gap_s,
  b_mean_diff_between_eit_gaps_s
  ]
  
  guided_baseline_data = [convergence_iterations, max_score_iterations, convergence_scores, max_scores, execution_times, presence_rates, all_vectors_coverages, top_vectors_coverages, popularities, constraints_completions, potentials, links_gaps, vector_links_gaps, ee_powers,
  min_eit_gap_s,
  max_eit_gap_s,
  diff_min_max_eit_gaps_s,
  mean_eit_gap_s,
  mean_diff_between_eit_gaps_s
  ]
  
  titles = ['convergence_iterations', 'max_score_iterations', 'convergence_scores', 'max_scores', 'execution_times', 'presence_rates', 'all_vectors_coverages', 'top_vectors_coverages', 'popularities', 'constraints_completions', 'potentials', 'links_gaps', 'vector_links_gaps', 'ee_powers', 
  'min_eit_gap_s', 'max_eit_gap_s', 'diff_min_max_eit_gaps_s', 'mean_eit_gap_s', 'mean_diff_between_eit_gaps_s'
  ]



  
  
  if algo_name != "Genetic" and algo_name != "Reinforcement":
    
    # Do all statistical tests with baseline results and current algorithm results.
    statest_text_results = []
    for k in range(len(baseline_data)):
      # First, we’ll create two arrays to hold the values for both baseline and current algorithm.
      group1 = baseline_data[k]
      group2 = guided_baseline_data[k]
      # Next, we’ll use the ranksums() function from the scipy.stats library to conduct the statistical test, which uses the following syntax:
      # ranksums(x, y)
      # where:
      # x: an array of sample observations from group 1
      # y: an array of sample observations from group 2
      statest_results = ranksums(group1, group2)
      # The Wilcoxon rank-sum test tests the null hypothesis that two sets of measurements are drawn from the same distribution. The alternative hypothesis is that values in one sample are more likely to be larger than the values in the other sample.
      # This test should be used to compare two samples from continuous distributions. It does not handle ties between measurements in x and y. For tie-handling and an optional continuity correction see scipy.stats.mannwhitneyu.
      # Hypothesis :
      #   H0: The two sets of measurements are drawn from the same distribution.
      #   HA: Values in one sample are more likely to be larger than the values in the other sample.
      # If p-value >= 0.05, H0 is not rejected. 
      # If p-value < 0.05, H0 is rejected.
      p_value = statest_results[1]
      if p_value >= 0.05:
        statest_text_results.append("Fail")
      else:
        statest_text_results.append("Pass")

    # Save boxplots of each evaluation criterion results. Each chart will contain the boxplot of baseline and the boxplot of the current guiding function for a specific evaluation criterion.
    for k in range(len(baseline_data)):
      data = [baseline_data[k], guided_baseline_data[k]]
      plt.boxplot(data)
      plt.xlabel('Algorithm')
      plt.ylabel(titles[k])
      plt.xticks([1, 2], ["Baseline", algo_name]) # each x-axis labels
      plt.savefig('Experiments/Charts/Boxplots/' + titles[k] + '_id' + str(id) + '_' + algo_name + '.png', dpi=300, bbox_inches='tight')
      #plt.show()
      plt.clf()

  
  """
  # Display the evolution of :
  # - max scores iterations # red dashes : 'r--'
  # - convergences iterations # blue dashes : 'b--'
  plt.plot(
            [*range(0, macro_iterations, 1)], max_score_iterations, 'r-',
            [*range(0, macro_iterations, 1)], convergence_iterations, 'b-'
          )
  red_line = mlines.Line2D([], [], color='red', label='Iterations To Reach Max Score')
  blue_line = mlines.Line2D([], [], color='blue', label='Iterations To Reach Convergence')
  mean_1 = mlines.Line2D([], [], color='black', label='Mean Iterations To Reach Max Score = '+str(mean_msi))
  mean_2 = mlines.Line2D([], [], color='black', label='Mean Iterations To Reach Convergence = '+str(mean_ci))
  mean_3 = mlines.Line2D([], [], color='black', label='Mean Execution Time = '+str(mean_et))
  entropy_1 = mlines.Line2D([], [], color='black', label='Entropy Iterations To Reach Max Score = '+entropy_msi)
  entropy_2 = mlines.Line2D([], [], color='black', label='Entropy Iterations To Reach Convergence = '+entropy_ci)
  entropy_3 = mlines.Line2D([], [], color='black', label='Entropy Execution Time = '+entropy_et)
  plt.legend(handles=[red_line, blue_line, mean_1, mean_2, mean_3])
  plt.xlabel('Macro Iteration')
  plt.ylabel('Algorithm Iterations')
  plt.savefig('Experiments/Charts/'+algo_name+'_id_' + str(id) + '_Iterations_Stag'+str(stagnation)+'.png', dpi=300, bbox_inches='tight')
  #plt.show()
  plt.clf()

  # Display the evolution of :
  # - max scores # red dashes : 'r--'
  # - convergences scores # blue dashes : 'b--'
  # - real mean scores # green dashes : 'g--'
  plt.plot(
            [*range(0, macro_iterations, 1)], max_scores, 'r-',
            [*range(0, macro_iterations, 1)], convergence_scores, 'b-',
            #[*range(0, macro_iterations, 1)], real_mean_scores, 'g-'
          )
  red_line = mlines.Line2D([], [], color='red', label='Max Scores')
  blue_line = mlines.Line2D([], [], color='blue', label='Convergence Scores')
  #green_line = mlines.Line2D([], [], color='green', label='Real Scores')
  mean_1 = mlines.Line2D([], [], color='black', label='Mean Max Scores = '+str(mean_ms))
  mean_2 = mlines.Line2D([], [], color='black', label='Mean Convergence Scores = '+str(mean_cs))
  #mean_3 = mlines.Line2D([], [], color='black', label='Mean Real Scores = '+str(mean_rmscores))
  entropy_1 = mlines.Line2D([], [], color='black', label='Entropy Max Scores = '+entropy_ms)
  entropy_2 = mlines.Line2D([], [], color='black', label='Entropy Convergence Scores = '+entropy_cs)
  #entropy_3 = mlines.Line2D([], [], color='black', label='Entropy Real Scores = '+entropy_rmscores)
  plt.legend(handles=[red_line, blue_line, mean_1, mean_2])
  plt.xlabel('Macro Iteration')
  plt.ylabel('Score')
  plt.savefig('Experiments/Charts/'+algo_name+'_id_' + str(id) + '_Scores_Stag'+str(stagnation)+'.png', dpi=300, bbox_inches='tight')
  #plt.show()
  plt.clf()      
  
  # Display the distribution of mean_dense_vector and mean_presence_gap in order to visualize exploration and exploitation.
  digit_names = []
  for k in range(0, len(mean_dense_vector)): 
    digit_names.append(k)
    
  # mean_dense_vector bar chart.
  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  values = mean_dense_vector
  ax.bar(digit_names,values)
  plt.xlabel('Digit')
  plt.ylabel('Quantity')
  plt.savefig('Experiments/Charts/'+algo_name+'_id_' + str(id) + '_Presence_Dense_Stag'+str(stagnation)+'.png', dpi=300, bbox_inches='tight')
  #plt.show()
  plt.clf()
    
  # mean_presence_gap bar chart.
  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  values = mean_presence_gap
  ax.bar(digit_names,values)
  mean_1 = mlines.Line2D([], [], color='black', label='Presence Rate = '+str(mean_presence_rate) + '%')
  plt.legend(handles=[mean_1])
  plt.xlabel('Digit')
  plt.ylabel('Mean Presence Gap VS Other Digits')
  plt.savefig('Experiments/Charts/'+algo_name+'_id_' + str(id) + '_Presence_Gap_Stag'+str(stagnation)+'.png', dpi=300, bbox_inches='tight')
  #plt.show()
  plt.clf() 
  
  
  # Display the evolution of :
  # - all_vectors coverage # red dashes : 'r--'
  # - top_vectors coverage # blue dashes : 'b--'
  plt.plot(
            [*range(0, macro_iterations, 1)], [x*100 for x in all_vectors_coverages], 'r-',
            [*range(0, macro_iterations, 1)], [x*100 for x in top_vectors_coverages], 'b-'
          )
  red_line = mlines.Line2D([], [], color='red', label='All Vectors Coverage')
  blue_line = mlines.Line2D([], [], color='blue', label='Top Vectors Coverage')
  mean_1 = mlines.Line2D([], [], color='black', label='Mean All Vectors Coverage = '+str(mean_avcov) + '%')
  mean_2 = mlines.Line2D([], [], color='black', label='Mean Top Vectors Coverage = '+str(mean_tvcov) + '%')
  plt.legend(handles=[red_line, blue_line, mean_1, mean_2])
  plt.xlabel('Macro Iteration')
  plt.ylabel('Coverage (%)')
  plt.savefig('Experiments/Charts/'+algo_name+'_id_' + str(id) + '_Coverage_Stag'+str(stagnation)+'.png', dpi=300, bbox_inches='tight')
  #plt.show()
  plt.clf() 
  
  
  # Display df_indexes_and_popularity with the head/tail separation.
  resource_indexes = []
  for k in range(0, len(df_indexes_and_popularity)): 
    resource_indexes.append(k)
  # bar chart.
  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  values = array(df_indexes_and_popularity)[:,1]
  ax.bar(resource_indexes,values)
  separation = mlines.Line2D([], [], color='red', label='Head/Tail Separation')
  mean_1 = mlines.Line2D([], [], color='black', label='Mean Popularity = '+str(mean_pop) + '%')
  plt.legend(handles=[separation, mean_1])
  plt.xlabel('Index')
  plt.ylabel('Popularity')
  # Draw a vertical line which separate the head from the tail using the percentage given as parameter.
  x_value = int(len(df_indexes_and_popularity)*popularity_head_percentage)
  plt.axvline(x=x_value, ymin=0, ymax=1000, label='Head/Tail Separation', color='red')
  # Save
  plt.savefig('Experiments/Charts/'+algo_name+'_id_' + str(id) + '_Popularity_Stag'+str(stagnation)+'.png', dpi=300, bbox_inches='tight')
  #plt.show()
  plt.clf()
  
  
  # Display the evolution of :
  # - constraints completions # green dashes : 'g--'
  plt.plot(
            [*range(0, macro_iterations, 1)], constraints_completions, 'g-'
          )
  green_line = mlines.Line2D([], [], color='green', label='Constraints Completion')
  mean_3 = mlines.Line2D([], [], color='black', label='Mean Constraints Completion = '+str(mean_const_comp) + '%')
  entropy_3 = mlines.Line2D([], [], color='black', label='Entropy Constraints Completion = '+entropy_const_comp)
  plt.legend(handles=[green_line, mean_3])
  plt.xlabel('Macro Iteration')
  plt.ylabel('Constraints Completion')
  plt.savefig('Experiments/Charts/'+algo_name+'_id_' + str(id) + '_Constraints Completion_Stag'+str(stagnation)+'.png', dpi=300, bbox_inches='tight')
  #plt.show()
  plt.clf()
  """
  
  
  """
  # -------------------
  # Display E&E Specter
  # -------------------
  # Remove duplicates from all_vectors.
  all_vectors_cleaned = []
  for k in range(len(all_vectors)):
    if all_vectors[k] not in all_vectors_cleaned:
      all_vectors_cleaned.append(all_vectors[k])
  # --------------------
  # Save score and constraints completion score of each vector from all_vectors_cleaned.
  av_clean_scores = []
  av_clean_cc_scores = []
  for k in range(len(all_vectors_cleaned)):
    v_score = score(all_vectors_cleaned[k])
    av_clean_scores.append(v_score)
    v_cc_score = vector_constraints_score(all_vectors_cleaned[k])
    av_clean_cc_scores.append(round(v_cc_score*100, 1))
  # Display all_vectors E&E Specter (scores)
  fig = plt.figure(figsize=(8,5))
  ax = fig.add_axes([0.1, 0.1, 0.6, 0.7])
  fig.suptitle('P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_E&E_Specter', fontsize=12)
  plt.title('All Vectors E&E Specter - Fitness & CC Score', fontsize=10)
  plot_range = av_clean_scores
  ax.plot(plot_range, av_clean_cc_scores, 'b.', linewidth=0.5, markersize=1, 
            label='E&E Specter')
  ax.set_xlabel('Fitness Score', size=10)
  ax.set_ylabel('CC Score (%)', size=10)
  ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.savefig('Experiments/Charts/Specter/P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_E&E_Specter.png', dpi=300, bbox_inches='tight')
  #plt.show() 
  plt.clf()
  # Close all figures.
  plt.close('all')
  # --------------------
  # Save HVV and HVO of each vector from all_vectors_cleaned.
  av_clean_hvvs = []
  av_clean_hvos = []
  for k in range(len(all_vectors_cleaned)):
    # Get the pool of 1-digits resources of the vector.
    pool = []
    for d in range(len(all_vectors_cleaned[k])):
      if all_vectors_cleaned[k][d] == 1:
        pool.append(df[d])
    # Calculate the vector hvv
    hvvd = 1
    pool_maxes = np_max(array(pool), axis=0)
    for pm in range(1, len(pool_maxes)):
      hvvd *= pool_maxes[pm]
    hvv = hvvd / hvv_max
    av_clean_hvvs.append(hvv)
    # Calculate the vector hvo
    hvod = 1
    pool_matrix = np_matrix(pool)
    pool_transposed = pool_matrix.T
    for e in range(1, len(pool_transposed)):
      # Get the unique values of the dimension considered by converting the list to a set. Calculate its len and multiply HVOd by it.
      qty = len(set(pool_transposed[e].tolist()[0]))
      hvod *= qty
    hvo = hvod / hvo_max
    av_clean_hvos.append(hvo)
  # Display all_vectors E&E Specter (HVs)
  fig = plt.figure(figsize=(8,5))
  ax = fig.add_axes([0.1, 0.1, 0.6, 0.7])
  fig.suptitle('P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_E&E_Specter', fontsize=12)
  plt.title('All Vectors E&E Specter - HVV & HVO', fontsize=10)
  plot_range = av_clean_hvvs
  ax.plot(plot_range, av_clean_hvos, 'b.', linewidth=0.5, markersize=1, 
            label='E&E Specter')
  ax.set_xlabel('HVV', size=10)
  ax.set_ylabel('HVO', size=10)
  ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.savefig('Experiments/Charts/Specter/P' + str(profile_number) + '_' + current_algo + '_' + constraints_variation_name + '_Stag' + str(stagnation) + '_E&E_Specter.png', dpi=300, bbox_inches='tight')
  #plt.show() 
  plt.clf()  
  # Close all figures.
  plt.close('all')
  """








  
  # Save results in excel file.
  worksheet.write('A'+str(row_spacer), algo_name) 
  worksheet.write('B'+str(row_spacer), reshape_counter)
  worksheet.write('C'+str(row_spacer), df_final_size)
  worksheet.write('D'+str(row_spacer), mean_ci) 
  worksheet.write('E'+str(row_spacer), mean_msi) 
  worksheet.write('F'+str(row_spacer), mean_cs)
  worksheet.write('G'+str(row_spacer), mean_ms)
  worksheet.write('H'+str(row_spacer), mean_et)
  #worksheet.write('G'+str(row_spacer), mean_rmsim)
  #worksheet.write('G'+str(row_spacer), mean_rmscores)
  worksheet.write('I'+str(row_spacer), mean_presence_rate)
  worksheet.write('J'+str(row_spacer), mean_avcov)
  worksheet.write('K'+str(row_spacer), mean_tvcov)
  worksheet.write('L'+str(row_spacer), mean_pop)
  worksheet.write('M'+str(row_spacer), mean_const_comp)
  worksheet.write('N'+str(row_spacer), mean_potential)
  worksheet.write('O'+str(row_spacer), mean_links_gaps)
  worksheet.write('P'+str(row_spacer), mean_vector_links_gaps)
  
  worksheet.write('Q'+str(row_spacer), mean_ee_powers)
  #worksheet.write('P'+str(row_spacer), mean_counters_explore_iter)
  #worksheet.write('Q'+str(row_spacer), mean_counters_exploit_iter)
  #worksheet.write('R'+str(row_spacer), mean_counters_explore_vector)
  #worksheet.write('S'+str(row_spacer), mean_counters_exploit_vector)
  #worksheet.write('T'+str(row_spacer), mean_counters_explore_digit)
  #worksheet.write('U'+str(row_spacer), mean_counters_exploit_digit)
  
  # If not enough eit, gap values are inf so we put them at 99999 to be able to print them in the worksheet.
  if math.isinf(mean_of_min_eit_gap_s):
    mean_of_min_eit_gap_s = 99999
  if math.isinf(mean_of_max_eit_gap_s):
    mean_of_max_eit_gap_s = 99999
  if math.isinf(mean_of_diff_min_max_eit_gaps_s):
    mean_of_diff_min_max_eit_gaps_s = 99999
  if math.isinf(mean_of_mean_eit_gap_s):
    mean_of_mean_eit_gap_s = 99999
  if math.isinf(mean_of_mean_diff_between_eit_gaps_s):
    mean_of_mean_diff_between_eit_gaps_s = 99999
  
  worksheet.write('R'+str(row_spacer), mean_of_min_eit_gap_s)
  worksheet.write('S'+str(row_spacer), mean_of_max_eit_gap_s)
  worksheet.write('T'+str(row_spacer), mean_of_diff_min_max_eit_gaps_s)
  worksheet.write('U'+str(row_spacer), mean_of_mean_eit_gap_s)
  worksheet.write('V'+str(row_spacer), mean_of_mean_diff_between_eit_gaps_s)  
  
  
  if algo_name != "Genetic" and algo_name != "Reinforcement":
    # Statistical Test line.
    row_spacer += 1
    worksheet.write('A'+str(row_spacer), "Statistical Test VS Baseline") 
    worksheet.write('B'+str(row_spacer), str(""))
    worksheet.write('C'+str(row_spacer), str(""))
    worksheet.write('D'+str(row_spacer), str(statest_text_results[0])) 
    worksheet.write('E'+str(row_spacer), str(statest_text_results[1])) 
    worksheet.write('F'+str(row_spacer), str(statest_text_results[2]))
    worksheet.write('G'+str(row_spacer), str(statest_text_results[3]))
    worksheet.write('H'+str(row_spacer), str(statest_text_results[4]))
    #worksheet.write('G'+str(row_spacer), str(statest_text_results[0]))
    #worksheet.write('G'+str(row_spacer), str(statest_text_results[0]))
    worksheet.write('I'+str(row_spacer), str(statest_text_results[5]))
    worksheet.write('J'+str(row_spacer), str(statest_text_results[6]))
    worksheet.write('K'+str(row_spacer), str(statest_text_results[7]))
    worksheet.write('L'+str(row_spacer), str(statest_text_results[8]))
    worksheet.write('M'+str(row_spacer), str(statest_text_results[9]))
    worksheet.write('N'+str(row_spacer), str(statest_text_results[10]))
    worksheet.write('O'+str(row_spacer), str(statest_text_results[11]))
    worksheet.write('P'+str(row_spacer), str(statest_text_results[12]))
    
    worksheet.write('Q'+str(row_spacer), str(statest_text_results[13]))
    
    worksheet.write('R'+str(row_spacer), str(statest_text_results[14]))
    worksheet.write('S'+str(row_spacer), str(statest_text_results[15]))
    worksheet.write('T'+str(row_spacer), str(statest_text_results[16]))
    worksheet.write('U'+str(row_spacer), str(statest_text_results[17]))
    worksheet.write('V'+str(row_spacer), str(statest_text_results[18]))

  row_spacer += 1
  
  

# Close the Excel file.
workbook.close()




















