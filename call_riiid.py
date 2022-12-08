import sys
import os
import sqlite3
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
import numpy as np
from numpy import savetxt
from copy import deepcopy
from icecream import ic
from numpy import transpose
from statistics import mean
import csv
from numpy import array
from numpy import delete
import math
from numpy import argsort as argsort
from past.builtins import execfile
execfile('reshape.py')


# Get parameters given for script execution.
# The first parameter is the profile number.
profile_number = sys.argv[1]

# Reshape functions.
# "none","msm","dsm","ivs","mgtcm","sk","shk","csk"
reshape_functions = ["csk"]

# Execute for each reshape function.
for rf in range(len(reshape_functions)):

  # The reshape function for this iteration.
  reshape_function = reshape_functions[rf]

  # Get Raw Data from the right profile file.
  # df must have as 1st column something to recommend (here it is a movie id) and as other columns various criteria defining the thing to recommend.
  df = []
  with open("Experiments/profile_"+profile_number+"_riiid.txt", newline='') as txtfile:
    df = list(csv.reader(txtfile, delimiter=';'))

  # Transform df columns into int or float.
  for row in range(0, len(df)):
    df[row][1] = float(df[row][1].replace(",", "."))
    #df[row][2] = int(df[row][2])
    df[row][2] = float(df[row][2].replace(",", "."))
    df[row][3] = float(df[row][3].replace(",", "."))

  # Transform df into a Dataframe.
  df = DataFrame(df)
  df = df.values

  # Scale of the criteria = Scale of df's columns except the first.
  # MinMaxScaler() scales values to have them laying between 0 and 1 included.
  scaler = preprocessing.MinMaxScaler()
  df[:,1:] = scaler.fit_transform(df[:,1:])


  # CONSTRAINTS
  # We specify here which columns of df are impacted by constraints and what are minimum and maximum values of these constraints.
  # Each final vector must respect each constraint. We check that by summing all the vector 1-digits values for the constraint column (column in df). The sum is then compared to max_value and min_value of the constraint (see functions.py). 
  # For example, in the knapsack problem, the volume is a constraint because each item has a volume and we can't exceed the maximum volume of the knapsack.
  # A constraint takes the form [column_in_df, max_value, min_value].
  # Column 1 in df is the avg_answered_correctly.
  # Column 2 in df is the avg_prior_question_elapsed_time.
  # Column 3 in df is the avg_prior_question_had_explanation.
  constraints_variation_name = "V3_2C" # FOR FILE SAVE
  constraints = []
  if profile_number == "1":
    constraints.append([1, 7, 0])
    #constraints.append([2, 6, 0])
    constraints.append([3, 5, 0])
  elif profile_number == "2":
    constraints.append([1, 9, 0])
    #constraints.append([2, 8, 0])
    constraints.append([3, 7, 0])
  elif profile_number == "3":
    constraints.append([1, 8, 0])
    #constraints.append([2, 7, 0])
    constraints.append([3, 6, 0])
  elif profile_number == "4": 
    constraints.append([1, 7, 0])
    #constraints.append([2, 8, 0])
    constraints.append([3, 9, 0])
  elif profile_number == "5": 
    constraints.append([1, 8, 0])
    #constraints.append([2, 8, 0])
    constraints.append([3, 8, 0])
  elif profile_number == "6": 
    constraints.append([1, 8, 0])
    #constraints.append([2, 7, 0])
    constraints.append([3, 6, 0])


  # Save constraints as a file.
  pd.DataFrame(constraints).to_csv("Experiments/constraints.csv", header=None, index=None)


  # MAX / MIN
  # Once df is scaled, we specify here if criteria/columns have to be maximised or minimised by filling an array with 'max' or 'min' values. 
  # if maximisation : criterion_value = criterion_value (no change)
  # if minimisation : criterion_value = 1 - criterion_value (so that the more the value is low, the more it participates to the score)
  # NOTA : 1st element in this array corresponds to df 1st criterion and so on.
  criteria_min_or_max = []
  if profile_number == "1":  
    criteria_min_or_max = ['max', 'min', 'min']
  elif profile_number == "2":
    criteria_min_or_max = ['min', 'max', 'max']
  elif profile_number == "3":
    criteria_min_or_max = ['min', 'min', 'min']
  elif profile_number == "4":  
    criteria_min_or_max = ['max', 'min', 'max']
  elif profile_number == "5":  
    criteria_min_or_max = ['min', 'max', 'min']
  elif profile_number == "6":  
    criteria_min_or_max = ['max', 'max', 'max']

  # Transform df criteria/columns regarding this array.
  for k in range(0, len(criteria_min_or_max)):
    if criteria_min_or_max[k] == "min":
      for e in range(0, len(df)):
        df[e][k+1] = 1 - df[e][k+1]
        """
        if df[e][k+1] < 0:
          df[e][k+1] = 0
        """

  
  # --------------------
  # RESHAPE SEARCH SPACE
  # --------------------
  initial_df = deepcopy(df)
  df_initial_size = len(df)
  reshape_counter = 0

  # Do the reshape while df size is above a certain percentage of its initial size.
  # If the reshape_function is "none", reshape is not done.
  stop = False
  df_percent = 0.3
  while len(df) > (df_percent*df_initial_size) and stop is False:
    former_df = deepcopy(df)
    former_df_len = len(df)
    reshape_counter += 1
    #print("RESHAPE #" + str(reshape_counter))
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
    if len(df) == former_df_len:
      stop = True
    if len(df) < 0.15*df_initial_size:
      stop = True
      df = deepcopy(former_df)
      reshape_counter -= 1
      print("BACK TO FORMER DF !")
      
  df_final_size = len(df)
  if df_final_size == df_initial_size:
    reshape_counter = 0
  print("NUMBER OF RESHAPES : " + str(reshape_counter))
  print("DF SIZE BEFORE/AFTER RESHAPE : " + str(df_initial_size) + "/" + str(df_final_size))
  # --------------------
  
  
  # Round df values.
  for k in range(len(df)):
    for c in range(1, len(df[k])):
      df[k][c] = round(df[k][c], 3)

  # Save df as a file.
  pd.DataFrame(df).to_csv("Experiments/df.csv", header=None, index=None)


  # Popularity
  # Create a variable equal to df sorted by popularity DESC.
  # In our case, we represent popularity by the rating quantity (index 2 in df).
  df_indexes_and_popularity = []
  for k in range(len(df)):
    df_indexes_and_popularity.append({
                                      'index_in_df':k,
                                      'popularity':df[k][2]
                                  })
  # Sort df_indexes_and_popularity by popularity DESC.
  df_indexes_and_popularity = sorted(deepcopy(df_indexes_and_popularity), key=lambda dct: dct['popularity'], reverse=True)
  # Save df_indexes_and_popularity as a file.
  pd.DataFrame(df_indexes_and_popularity).to_csv("Experiments/df_indexes_and_popularity.csv", header=None, index=None)


  # Parameters to call main.py
  # Path to df.csv
  path_to_df = "Experiments/df.csv"
  # Macro iterations.
  macro_iterations = 30
  # Algorithm iterations.
  algorithm_iterations = 1000
  # Iterations needed without an evolution of max score to say that convergence is reached.
  stagnation = 100
  # This is the number of actors concerned by the recommendation. Each actor get a vector as recommendation.
  actors_number = 1
  # This is the minimum value of similarity between recommended vectors of different actors.
  similarity_percentage = 0.7 
  # Used for similarity. This is the max number of vectors in top_vectors used to create the best combination of vectors regarding similarity.
  max_top_vectors = 10
  # Algorithms to execute. Also used for files saved.
  # Type inside brackets the names of the algorithms you want to execute. 
  # Available algorithms : "Random", "Genetic", "Ramify", "Struggle", "Spycle", "Spasm", "Extossom", "U3S", "EEP", "SPEA2", "DEEP", "DE", "EDA", "DE_EDA", "DCEA"
  # Add a suffix "_ee_guided" to an algorithm to execute it with guided exploration and exploitation. => Do not support all algorithms.
  # Additional suffixes can be added to the genetic algorithm to use different guiding functions.
  """
  "Genetic_ee_guided_links"

  "Ramify", "Ramify_ee_guided", 
  "Struggle", "Struggle_ee_guided", 
  "Spycle", "Spycle_ee_guided", 
  "Spasm", "Spasm_ee_guided", 
  "Extossom", "Extossom_ee_guided"

  "Genetic", "Genetic_reshape"
  "Genetic_dynashape"
  
  "Genetic", "Genetic_ee_guided_separated","Genetic_ee_guided_merged", "Genetic_ee_guided_potential","Genetic_ee_guided_cc", "Genetic_ee_guided_vector_links", "Genetic_ee_guided_power", "Genetic_ee_guided_temporal_balance"
  
  "Genetic", "Ramify", "Struggle", "Spycle", "Spasm", "Extossom", "U3S", "EEP"
  
  "Genetic_ee_logged", "Reinforcement_ee_logged", "Ramify_ee_logged","Struggle_ee_logged","Spycle_ee_logged","Spasm_ee_logged","Extossom_ee_logged","U3S_ee_logged","EEP_ee_logged"
  
  "Reinforcement", "Reinforcement_reshape", "Reinforcement_ee_guided_separated","Reinforcement_ee_guided_merged", "Reinforcement_ee_guided_potential","Reinforcement_ee_guided_cc", "Reinforcement_ee_guided_vector_links", "Reinforcement_ee_guided_power", "Reinforcement_ee_guided_temporal_balance"
  
  "Combi_reduc"
  
  "Reinforcement_ee_guided_cc", "Reinforcement_ee_guided_vector_links", "Reinforcement_ee_guided_power", "Reinforcement_ee_guided_temporal_balance"
  
  "Genetic", "Combi_reduc"
  
  "Genetic", "Reinforcement"
  """
  
  # If the reshape function is "none", we execute "Genetic" and/or other "Genetic_ee_guided_...". Else, the specific current reshape function is used on "Genetic" and we call the function "Genetic_reshape" in order to get the statistical tests done (see main.py).
  algos = []
  if reshape_function == "none":
    algos = ["Reinforcement"]
  else:
    algos = ["Reinforcement", "Reinforcement_ee_guided_separated","Reinforcement_ee_guided_merged", "Reinforcement_ee_guided_potential","Reinforcement_ee_guided_cc", "Reinforcement_ee_guided_vector_links", "Reinforcement_ee_guided_power", "Reinforcement_ee_guided_temporal_balance"]
  algo_types = ""
  for t in algos:
    algo_types += t + "-"
  algo_types = algo_types[:-1] # remove last character ("-")
  # Path to df_indexes_and_popularity.csv
  path_to_df_indexes_and_popularity = "Experiments/df_indexes_and_popularity.csv"
  # Percentage of best popularities of df_indexes_and_popularity to be considered as the head. The rest will be the tail.
  popularity_head_percentage = 0.25 # The best 25% will be the head.
  # Path to constraints.csv
  path_to_constraints = "Experiments/constraints.csv"

  # Get df indexes of popular items using popularity_head_percentage.
  popular_items_indexes = []
  max_index = int(len(df)*popularity_head_percentage)
  for k in range(0, max_index):
    popular_items_indexes.append(df_indexes_and_popularity[k]['index_in_df'])
  # Save popular_items_indexes as a file.
  pd.DataFrame(popular_items_indexes).to_csv("Experiments/popular_items_indexes.csv", header=None, index=None)
  # Path to popular_items_indexes.csv
  path_to_popular_items_indexes = "Experiments/popular_items_indexes.csv"


  

  # Call main.py with its parameters.
  os.system("main.py " + path_to_df + " " + str(macro_iterations) + " " + str(algorithm_iterations) + " " + str(stagnation) + " " + str(actors_number) + " " + str(similarity_percentage) + " " + str(max_top_vectors) + " " + str(algo_types) + " " + path_to_df_indexes_and_popularity + " " + str(popularity_head_percentage) + " " + path_to_popular_items_indexes + " " + path_to_constraints + " " + profile_number + " " + str(reshape_counter) + " " + str(df_final_size) + " " + reshape_function + " " + constraints_variation_name)





















