import sys
import os
import sqlite3
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from numpy import savetxt
from copy import deepcopy
from icecream import ic
from numpy import transpose
from statistics import mean


'''
DATA STRUCTURE
Name : objective id
Stakeholder : concerned stakeholder id
Quality : stakeholder type
Possible values : teacher, learner, author, publisher, institution
Status : objective status
Possible values : achieved, active, inactive
Priority : objective priority
Possible values : weak, average, strong

Targets : sub-objectives
  Form : 
        {
          (
            so_id, 
            so_target_value, 
            so_current_value, 
            cmin_so_target_value, 
            cmax_so_target_value, 
            timestamp_start, 
            cmin_start, 
            cmax_start, 
            timestamp_end, 
            cmin_end, 
            cmax_end, 
            duration, 
            cmin_duration, 
            cmax_duration, 
            order, 
            recommendation_sequence, 
            status
          )
        }
so = sub_objective
cmin, cmax : These are the maximum (cmax) and minimum (cmin) values ​​of certain attributes. Also called “constraints”.
order = priority of the sub-objective (1, 2, 3, ...)
status = (todo, doing, done)
A recommendation_sequence (rs) takes the form : {(action_verb, resource, timestamp_start, timestamp_end, duration, status)}.
'''

# Objective Class
class Objective:
  def __init__(self):
    self.id = 0
    self.name = ""
    self.stakeholder = ""
    self.quality = ""
    self.status = ""
    self.priority = ""
    self.highest_education = ""
    self.num_of_prev_attempts = ""
    self.targets = []

# Sub_objective Class
class Sub_objective:
  def __init__(self):
    self.id_sub_objective = ""
    self.so_name = ""
    self.so_target_value = ""
    self.so_current_value = ""
    self.cmin_so_target_value = ""
    self.cmax_so_target_value = ""
    self.timestamp_start = ""
    self.cmin_start = ""
    self.cmax_start = ""
    self.timestamp_end = ""
    self.cmin_end = ""
    self.cmax_end = ""
    self.duration = ""
    self.cmin_duration = ""
    self.cmax_duration = ""
    self.order_do = ""
    self.recommendations = []
    self.status = ""

# Function to get an objective and its associated sub-objectives.
def get_objective():
  o = Objective()
  
  c.execute('''
                SELECT id_objective, objective_name, stakeholder_name, stakeholder_quality, objective_status, objective_priority, highest_education, num_of_prev_attempts FROM w_objectives
                WHERE id_objective = ''' + id_o
            )
  
  conn.commit()
  
  df = DataFrame(c.fetchall(), columns=['id_objective', 'objective_name','stakeholder_name', 'stakeholder_quality','objective_status','objective_priority','highest_education','num_of_prev_attempts'])

  for index, row in df.iterrows():
    o.id = row['id_objective']
    o.name = row['objective_name']
    o.stakeholder = row['stakeholder_name']
    o.quality = row['stakeholder_quality']
    o.status = row['objective_status']
    o.priority = row['objective_priority']
    o.highest_education = row['highest_education']
    o.num_of_prev_attempts = row['num_of_prev_attempts']


  c.execute('''
                SELECT id_sub_objective, so_name, so_target_value, so_current_value, cmin_so_target_value, cmax_so_target_value, timestamp_start, cmin_start, cmax_start, timestamp_end, cmin_end, cmax_end, duration, cmin_duration, cmax_duration, order_do, status, date_submit 
                FROM w_sub_objectives
                WHERE id_objective = ''' + id_o
            )
  
  conn.commit()
  
  df = DataFrame(c.fetchall(), columns=['id_sub_objective', 'so_name', 'so_target_value', 'so_current_value', 'cmin_so_target_value', 'cmax_so_target_value', 'timestamp_start', 'cmin_start', 'cmax_start', 'timestamp_end', 'cmin_end', 'cmax_end', 'duration', 'cmin_duration', 'cmax_duration', 'order_do', 'status', 'date_submit'])

  for index, row in df.iterrows():
    o.targets.append({
                        'id_sub_objective':row['id_sub_objective'],
                        'so_name':row['so_name'],
                        'so_target_value':row['so_target_value'],
                        'so_current_value':row['so_current_value'],
                        'cmin_so_target_value':row['cmin_so_target_value'],
                        'cmax_so_target_value':row['cmax_so_target_value'],
                        'timestamp_start':row['timestamp_start'],
                        'cmin_start':row['cmin_start'],
                        'cmax_start':row['cmax_start'],
                        'timestamp_end':row['timestamp_end'],
                        'cmin_end':row['cmin_end'],
                        'cmax_end':row['cmax_end'],
                        'duration':row['duration'],
                        'cmin_duration':row['cmin_duration'],
                        'cmax_duration':row['cmax_duration'],
                        'order_do':row['order_do'],
                        'recommendations':[],
                        'status':row['status']
                    })

  return o
  
# Get parameters given for script execution.
# The first parameter is the id of the objective.
id_o = sys.argv[1]
#param2 = sys.argv[2] # Other parameter if needed
#param3 = sys.argv[3] # Other parameter if needed
    
# Database Connection.
conn = sqlite3.connect('../oulad', isolation_level=None)
c = conn.cursor()
# Format of fields we get. "string" here.
conn.text_factory = str

# Get objective o from database.
o = get_objective()
# ic(o) # Print o

# For each sub-objective of o.
for so_o in o.targets:
  if len(so_o['recommendations']) == 0:
    
    # Get Raw Data For Multi-Objective Problem.
    # Get some values of achieved and inactive objectives.
    # As a consequence, inactive objectives will count in the calculation of resources scores. So, it takes into account the fact that a student want to succeed and fail at the same time. It also get other variables representing other objectives. Their values are minimized or maximized and summed inside the fitness function to represent the multi-objective score.
    # So here are the objectives :
    #   "Student success" by taking "achieved" objectives (current_value + quantity)
    #   "Student failure" by taking "inactive" objectives (current_value + quantity)
    #   "Student speed" by taking the number of remaining days before assessment end when the student submit his work.
    #   "Global representation of resources" by taking the quantity.
    #   "Students abilities" by taking the highest_education and the num_of_prev_attempts.
    #   "Resources popularity" by taking the sum_click.
    
    c.execute('''
              SELECT 
                
                w_recommendations.action_verb || '-' || w_recommendations.resource as action_resource,
                
                AVG(w_sub_objectives.so_current_value) as mean_so_current_value,
                
                COUNT(*) as quantity,
                
                AVG(w_sub_objectives.timestamp_end - w_sub_objectives.date_submit) as mean_remaining_days,
                
                AVG(highest_education) as highest_education,
                
                AVG(num_of_prev_attempts) as num_of_prev_attempts,
                
                AVG(sum_click) as sum_click
                
                --AVG(w_sub_objectives.so_target_value) as mean_so_target_value,
                --AVG(w_sub_objectives.duration) as mean_so_duration,
                --AVG(w_recommendations.duration) as mean_reco_duration
                
              FROM w_objectives
              INNER JOIN w_sub_objectives
              ON w_objectives.id_objective = w_sub_objectives.id_objective
              INNER JOIN w_recommendations
              ON w_sub_objectives.id_sub_objective = w_recommendations.id_sub_objective
              
              WHERE w_sub_objectives.so_name = "''' + so_o['so_name'] + '''"
              AND w_objectives.stakeholder_quality = "''' + o.quality + '''"
              GROUP BY resource, action_verb
              ''') 
    
    conn.commit()

    # df must have as 1st column something to recommend (here it is a pair action-resource) and as other columns various criteria defining the thing to recommend.
    df = DataFrame(c.fetchall(), columns=['action_resource', 'mean_so_current_value', 'quantity', 'mean_remaining_days', 'highest_education', 'num_of_prev_attempts', 'sum_click'])
    df = df.values

    # Scale of the criteria = Scale of df's columns except the first corresponding to the pairs action-resource.
    # MinMaxScaler() scales values to have them laying between 0 and 1 included.
    scaler = preprocessing.MinMaxScaler()
    df[:,1:] = scaler.fit_transform(df[:,1:])
    
    # CONSTRAINTS
    # Once df is scaled, we specify here which columns of df are impacted by constraints and what are minimum and maximum values of these constraints.
    # Each final vector must respect each constraint. We check that by summing all the vector 1-digits values for the constraint column (column in df). The sum is then compared to max_value and min_value of the constraint (see functions.py). 
    # For example, in the knapsack problem, the volume is a constraint because each item has a volume and we can't exceed the maximum volume of the knapsack.
    # Here, we consider quantity and sum_click as constraints. This is just one way to go, we could add or remove constraints, this depends on your application context.
    """
    # Evaluations of df desired constraints columns to determine constraints min and max values.
    transposed_df = transpose(df)
    ic(mean(transposed_df[2]))
    ic(mean(transposed_df[6]))
    ic(max(transposed_df[2]))
    ic(max(transposed_df[6]))
    ic(min(transposed_df[2]))
    ic(min(transposed_df[6]))
    sys.exit()
    """
    constraints = []
    constraints.append({
                          'column_in_df': 2, # Quantity column
                          'max_value': 1,
                          'min_value': 0.01
                      })
    constraints.append({
                          'column_in_df': 6, # sum_click column
                          'max_value': 1,
                          'min_value': 0.01
                      })
    # Save constraints as a file.
    pd.DataFrame(constraints).to_csv("Experiments/constraints.csv", header=None, index=None)
    
    # MAX / MIN
    # Once df is scaled, we specify here if criteria/columns have to be maximised or minimised by filling an array with 'max' or 'min' values. 
    # if maximisation : criterion_value = criterion_value (no change)
    # if minimisation : criterion_value = 1 - criterion_value (so that the more the value is low, the more it participates to the score)
    # NOTA : 1st element in this array corresponds to df 1st criterion and so on.
    criteria_min_or_max = ['max', 'max', 'max', 'min', 'min', 'max']
    # Transform df criteria/columns regarding this array.
    for k in range(0, len(criteria_min_or_max)):
      if criteria_min_or_max[k] == "min":
        for e in range(0, len(df)):
          df[e][k+1] = 1 - df[e][k+1]

    # Save df as a file.
    pd.DataFrame(df).to_csv("Experiments/df.csv", header=None, index=None)
    
    # Popularity
    # Create a variable equal to df sorted by popularity DESC.
    # In our case, we represent popularity by the quantity value (index 2 in df) because we consider that the more occurrences we have on a resource, the more this resource is popular.
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
    macro_iterations = 40
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
    "Random", "Random_ee_guided", 
    "Genetic", "Genetic_ee_guided_separated","Genetic_ee_guided_merged", "Genetic_ee_guided_potential","Genetic_ee_guided_cc", "Genetic_ee_guided_links", "Genetic_ee_guided_vector_links", "Genetic_ee_guided_power"
    "Ramify", "Ramify_ee_guided", 
    "Struggle", "Struggle_ee_guided", 
    "Spycle", "Spycle_ee_guided", 
    "Spasm", "Spasm_ee_guided", 
    "Extossom", "Extossom_ee_guided"
    """
    algos = ["Ramify", "Struggle", "Spycle", "Spasm", "Extossom", "U3S"]
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
    os.system("main.py " + path_to_df + " " + str(macro_iterations) + " " + str(algorithm_iterations) + " " + str(stagnation) + " " + str(actors_number) + " " + str(similarity_percentage) + " " + str(max_top_vectors) + " " + str(algo_types) + " " + path_to_df_indexes_and_popularity + " " + str(popularity_head_percentage) + " " + path_to_popular_items_indexes + " " + path_to_constraints + " " + id_o)


  # BREAK to do all of it only on one sub-objective.
  # Remove if you want to perform on all sub-objectives of the objective.
  break







