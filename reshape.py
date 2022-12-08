# --------------------
# Reshaping Functions
# Used to reshape the search space (df)
# --------------------


# Reshape with mean spp by keeping the resources from df that are above the mean of all mean spp.
def reshape_mean_spp_mean(): 
  global df
  
  # All spp values to determine the mean.
  all_spp = []
  
  tuples_with_mean_spp = []
  # For each resource in df.
  for k in range(len(df)):
    tuple_spp_values = []
    # Calculate the resource score.
    r_score = 0
    for c in range(1, len(df[k])):
      r_score += df[k][c]
    # For each column of the tuple.
    for c in range(len(df[k])):
      # Check if the column is under constraints.
      under_c = False
      for const in range(len(constraints)):
        if constraints[const][0] == c:      
          under_c = True
      if under_c is True:
        # Take the constraint max_value.
        c_max_val = 0
        for const in range(len(constraints)):
          if constraints[const][0] == c:
            c_max_val = constraints[const][1]
        # Calculate "spp" as the global score that a 100% constraints completion would give for this column. This means that for this column, 100% constraints completion is worth "spp" global score units.
        # For example : column cc = 12%, resource score = 7
        # 100% cc would give a score of : 
        #   (100*r_score)/(100*column score/column max constraint)
        # Or :
        #   r_score/(column score/column max constraint)
        spp = 0
        if df[k][c] == 0:
          spp = math.inf # Infinite number because division by zero.
        else:
          spp = r_score/(df[k][c]/c_max_val)
        tuple_spp_values.append(spp)
        
    # Calculate the mean spp of the tuple.
    mean_spp = mean(tuple_spp_values)
    # Associate it to the tuple.
    tuples_with_mean_spp.append({
                                  'tuple':df[k],
                                  'mean_spp':mean_spp
                                })
    # Add the mean_spp to all_spp.
    all_spp.append(mean_spp)
  
  # In theory, the more a tuple has a high mean_spp, the more it is good because it maximizes the score that a 100% constraints completion would give. So we should keep these to create the new df.
  
  # Replace inf values in all_spp with the max value other than 'inf' in all_spp. We do that to get an exploitable mean value.
  all_spp_len = len(all_spp)
  k = 0
  counter_delete = 0
  while k < all_spp_len:
    if math.isinf(float(all_spp[k])):
      all_spp.pop(k)
      counter_delete += 1
      k -= 1
      all_spp_len -= 1
    k += 1
  all_spp_max = max(all_spp)
  for k in range(counter_delete):
    all_spp.append(all_spp_max)
  
  # Calculate the global mean spp. 
  global_mean_spp = mean(all_spp)
  
  # Recreate df with the tuples having a spp...
  df = []
  
  # "KINGS"
  # ...superior to the global mean spp.
  for k in range(len(tuples_with_mean_spp)):
    if tuples_with_mean_spp[k]['mean_spp'] >= global_mean_spp:
      df.append(tuples_with_mean_spp[k]['tuple'])
  """
  # "RATS"
  # ...inferior to the global mean spp.
  for k in range(len(tuples_with_mean_spp)):
    if tuples_with_mean_spp[k]['mean_spp'] <= global_mean_spp:
      df.append(tuples_with_mean_spp[k]['tuple'])
  """
  
  """
  # If there are less than 1/5 of the original resources in df, we reset df and feed it with the 1/5 best resources from tuples_with_mean_spp regarding mean_spp.
  if len(df) < int(len(tuples_with_mean_spp)/5):
    df = []
    # Sort tuples_with_mean_spp by mean_spp DESC.
    tuples_with_mean_spp = sorted(deepcopy(tuples_with_mean_spp), key=lambda dct: dct['mean_spp'], reverse=True)
    # Feed df.
    for k in range(int(len(tuples_with_mean_spp)/5)):
      df.append(tuples_with_mean_spp[k]['tuple'])
    ic("< 1/5 of original resources")
  else:
    ic(">= 1/5 of original resources")
  """
  
  df = DataFrame(df)
  df = df.values


# Reshape with detailed spp by keeping the resources from df that are above the mean of all detailed spp.
def reshape_detailed_spp_mean(): 
  global df
  
  final_tuples = []
  
  # Change the structure of df.
  df_restructured = []
  for k in range(len(df)):
    df_restructured.append({
                              'tuple':df[k],
                              'score':0,
                              'spp':0
                          })
  df = []
  df = df_restructured
  
  # Calculate the score of each df tuple.
  for k in range(len(df)):
    # Calculate the resource score.
    r_score = 0
    for c in range(1, len(df[k]['tuple'])):
      r_score += df[k]['tuple'][c]
    df[k]['score'] = r_score
  
  # For each column.
  for c in range(len(df[0]['tuple'])):
  
    # All spp values to determine the mean.
    all_spp = []
  
    # Check if the column is under constraints.
    under_c = False
    for const in range(len(constraints)):
      if constraints[const][0] == c:      
        under_c = True
    if under_c is True:    
      
      # Take the constraint max_value.
      c_max_val = 0
      for const in range(len(constraints)):
        if constraints[const][0] == c:
          c_max_val = constraints[const][1]    
    
      # For each resource in df.
      for k in range(len(df)):
        # Calculate "spp" as the global score that a 100% constraints completion would give for this column. This means that for this column, 100% constraints completion is worth "spp" global score units.
        # For example : column cc = 12%, resource score = 7
        # 100% cc would give a score of : 
        #   (100*r_score)/(100*column score/column max constraint)
        # Or :
        #   r_score/(column score/column max constraint)
        spp = 0
        if df[k]['tuple'][c] == 0:
          spp = math.inf # Infinite number because division by zero.
        else:
          spp = df[k]['score']/(df[k]['tuple'][c]/c_max_val)
        df[k]['spp'] = spp
        
        all_spp.append(spp)

      # In theory, the more a tuple has a high mean_spp, the more it is good because it maximizes the score that a 100% constraints completion would give. So we should keep these to create the new df.

      # Replace inf values in all_spp with the max value other than 'inf' in all_spp. We do that to get an exploitable mean value.
      all_spp_len = len(all_spp)
      k = 0
      counter_delete = 0
      while k < all_spp_len:
        if math.isinf(float(all_spp[k])):
          all_spp.pop(k)
          counter_delete += 1
          k -= 1
          all_spp_len -= 1
        k += 1
      all_spp_max = max(all_spp)
      for k in range(counter_delete):
        all_spp.append(all_spp_max)
      
      # Calculate the global mean spp. 
      global_mean_spp = mean(all_spp)

      # Put in final_tuples the tuples from df having a spp...
      for k in range(len(df)):
        
        # "KINGS"
        # ...superior to the global mean spp.
        if df[k]['spp'] >= global_mean_spp:
          # Check if df[k]['tuple'] is not already present in final_tuples.
          check = True
          for t in range(len(final_tuples)):
            if list(final_tuples[t]) == list(df[k]['tuple']):
              check = False
          # Add in final_tuples.
          if check is True:
            final_tuples.append(df[k]['tuple'])
        """
        # "RATS"
        # ...inferior to the global mean spp.
        if df[k]['spp'] < global_mean_spp:
          # Check if df[k]['tuple'] is not already present in final_tuples.
          check = True
          for t in range(len(final_tuples)):
            if list(final_tuples[t]) == list(df[k]['tuple']):
              check = False
          # Add in final_tuples.
          if check is True:
            final_tuples.append(df[k]['tuple'])       
        """
      
  # Recreate df as a copy of final_tuples.
  df = []
  df = final_tuples
  df = DataFrame(df)
  df = df.values


# Reshape with tuples intra-values stability (ivs), keeping the resources from df that are the more/less stable regarding ivs.
def reshape_intra_values_stability(): 
  global df

  final_tuples = []
  all_ivs = []
  
  # Change the structure of df.
  df_restructured = []
  for k in range(len(df)):
    df_restructured.append({
                              'tuple':df[k],
                              'ivs':0
                          })
  df = []
  df = df_restructured

  # Calculate the ivs of each tuple which is equal to the mean value difference between column value pairs.
  for k in range(len(df)):
    gaps = []
    for c in range(1, len(df[k]['tuple'])):
      for c2 in range(c+1, len(df[k]['tuple'])):
        gap = abs(df[k]['tuple'][c] - df[k]['tuple'][c+1])
        gaps.append(gap)
    ivs = mean(gaps)
    df[k]['ivs'] = ivs
    all_ivs.append(ivs)
    
  # Calculate the mean of all ivs.
  mean_ivs = mean(all_ivs)
  
  # Put in final_tuples all tuples having a ivs...
  """
  # "RATS"
  # ...inferior to mean_ivs.
  for k in range(len(df)):
    if df[k]['ivs'] <= mean_ivs:
      final_tuples.append(df[k]['tuple'])
  """
  # "KINGS"
  # ...superior to mean_ivs.
  for k in range(len(df)):
    if df[k]['ivs'] >= mean_ivs:
      final_tuples.append(df[k]['tuple'])    
  
  # Recreate df as a copy of final_tuples.
  df = []
  df = final_tuples
  df = DataFrame(df)
  df = df.values
  


# Reshape by mean gap toward column means.
# 1/ For each column in df, we calculate the mean value (columns_means).
# 2/ For each tuple in df, we calculate the mean gap of the tuple column values toward the columns_means.
# 3/ Keep the resources from df having a mean gap superior/inferior to the mean of all tuples mean gaps.
def reshape_mean_gap_toward_columns_means():
  global df

  final_tuples = []
  
  # Transform df 1st column into integers.
  for k in range(len(df)):
    df[k][0] = int(df[k][0])
  
  # Create a vector named columns_means containing the mean value of each df columns.
  columns_means = array(df).mean(axis=0)
  
  # Calculate the mean_gap of each tuple and save them in final_tuples.
  # Example for a tuple : 
  #   1/ Take the first attribute and calculate its gap toward the column mean from columns_means.
  #   2/ Do 1/ for other attributes of the tuple.
  #   3/ Calculate mean_gap as the mean of all gaps obtained.
  #   4/ Save the tuple and its mean_gap into final_tuples.
  column_number = len(df[0])
  all_mean_gaps = []
  for k in range(len(df)):
    gaps = []
    for c in range(1, column_number):
      gap = abs(df[k][c] - columns_means[c])
      gaps.append(gap)
    mean_gap = mean(gaps)
    final_tuples.append({
                          'tuple':df[k],
                          'mean_gap':mean_gap
                        })
    all_mean_gaps.append(mean_gap)
  
  # Calculate the global mean gap.
  global_mean_gap = mean(all_mean_gaps)
  
  # Recreate df with the resources from final_tuples having a mean gap... 
  df = []
  
  # "KINGS"
  # ...superior to global_mean_gap.
  for t in range(len(final_tuples)):
    if final_tuples[t]['mean_gap'] > global_mean_gap:
      df.append(final_tuples[t]['tuple'])
  """    
  # "RATS"
  # ...inferior to global_mean_gap.
  for t in range(len(final_tuples)):
    if final_tuples[t]['mean_gap'] < global_mean_gap:
      df.append(final_tuples[t]['tuple'])  
  """
  df = DataFrame(df)
  df = df.values



# Resource Score
# Function to evaluate the score of a specific resource.
def resource_score_reshape(resource_index): 
  score = 0
  for column in range(1, len(df[0])):
    score += df[resource_index][column]
  return score

# Reshape df by keeping the best score resources.
def reshape_score_kings():
  global df
  final_tuples = []
  all_s = []

  # For each resource, calculate its score and save it in final_tuples.
  for k in range(len(df)):
    s = resource_score_reshape(k)
    all_s.append(s)
    final_tuples.append({
                          'tuple':df[k],
                          's':s
                        })
  
  # Calculate the mean_s.
  mean_s = mean(all_s)
  
  # Reset df.
  df = []
  """
  # "KINGS"
  # Recreate df with tuples having a s superior to mean_s.
  for t in range(len(final_tuples)):
    if final_tuples[t]['s'] > mean_s:
      df.append(final_tuples[t]['tuple'])
  """
  # "RATS"
  # Recreate df with tuples having a s inferior to mean_s.
  for t in range(len(final_tuples)):
    if final_tuples[t]['s'] < mean_s:
      df.append(final_tuples[t]['tuple'])
  
  df = DataFrame(df)
  df = df.values



# Reshape df by keeping the tuples respecting conditions toward various counters, considering each column (hill) separately.
# The counters are determined considering 5 aggregations :
#   - An array containing the mean value of each column (cm).
#   - An array containing the min value of each column (cmin).
#   - An array containing the max value of each column (cmax).
#   - An array containing the mean point (between cm and cmin) of each column (mpmin).
#   - An array containing the mean point (between cm and cmax) of each column (mpmax).
def reshape_score_hills_kings():
  global df
  final_tuples = []
  final_tuples_2 = []

  # Transform df 1st column into integers.
  for k in range(len(df)):
    df[k][0] = int(df[k][0])
    
  # Create a vector containing the mean values of df columns.
  columns_means = array(df).mean(axis=0)
  # Create a vector containing the min values of df columns.
  columns_mins = array(df).min(axis=0)
  # Create a vector containing the max values of df columns.
  columns_maxs = array(df).max(axis=0)
  # Create the mean points max and the mean points min.
  mean_points_max = []
  mean_points_min = []
  for k in range(len(columns_means)):
    mp_max = mean([columns_means[k], columns_maxs[k]])
    mean_points_max.append(mp_max)
    mp_min = mean([columns_means[k], columns_mins[k]])
    mean_points_min.append(mp_min)  
  
  # Put in final_tuples the tuples respecting conditions regarding the counters.
  number_of_columns = len(df[0])-1
  for t in range(len(df)):

    # Initialization of all counters.
    counter_above_cm = 0
    counter_below_cm = 0
    
    counter_above_mpmax = 0
    counter_below_mpmax = 0
    
    counter_above_mpmin = 0
    counter_below_mpmin = 0
    
    counter_above_cmin = 0
    counter_below_cmin = 0
    
    counter_above_cmax = 0
    counter_below_cmax = 0
    
    counter_between_cmin_cmax = 0
    counter_between_mpmin_mpmax = 0
    
    counter_between_cm_mpmax = 0
    counter_between_cm_mpmin = 0
    
    counter_between_cm_cmin = 0
    counter_between_cm_cmax = 0
    
    counter_between_mpmax_cmax = 0
    counter_between_mpmin_cmin = 0
    
    # Increment counters.
    # For each column...
    for c in range(1, len(df[0])):
      if df[t][c] > mean_points_max[c]:
        counter_above_mpmax += 1
      if df[t][c] < mean_points_max[c]:
        counter_below_mpmax += 1
      
      if df[t][c] > mean_points_min[c]:
        counter_above_mpmin += 1
      if df[t][c] < mean_points_min[c]:
        counter_below_mpmin += 1
      
      if df[t][c] > columns_mins[c]:
        counter_above_cmin += 1
      if df[t][c] < columns_mins[c]:
        counter_below_cmin += 1

      if df[t][c] > columns_maxs[c]:
        counter_above_cmax += 1
      if df[t][c] < columns_maxs[c]:
        counter_below_cmax += 1

      if df[t][c] > columns_means[c]:
        counter_above_cm += 1
      if df[t][c] < columns_means[c]:
        counter_below_cm += 1
    
      if df[t][c] > columns_mins[c] and df[t][c] < columns_maxs[c]:
        counter_between_cmin_cmax += 1
      if df[t][c] > mean_points_min[c] and df[t][c] < mean_points_max[c]:
        counter_between_mpmin_mpmax += 1
    
      if df[t][c] > columns_means[c] and df[t][c] < mean_points_max[c]:
        counter_between_cm_mpmax += 1
      if df[t][c] > mean_points_min[c] and df[t][c] < columns_means[c]:
        counter_between_cm_mpmin += 1    
    
      if df[t][c] > columns_mins[c] and df[t][c] < columns_means[c]:
        counter_between_cm_cmin += 1
      if df[t][c] > columns_means[c] and df[t][c] < columns_maxs[c]:
        counter_between_cm_cmax += 1     
   
      if df[t][c] > mean_points_max[c] and df[t][c] < columns_maxs[c]:
        counter_between_mpmax_cmax += 1
      if df[t][c] > columns_mins[c] and df[t][c] < mean_points_min[c]:
        counter_between_mpmin_cmin += 1      
    
    
    # Use conditions to decide if the tuple has to be kept.
    
    #if counter_above_cm == 0:                    
      # => NO DIGITS   
    #if counter_above_cm > 0:                     
      # => ALL DIGITS
    #if counter_above_cm == number_of_columns:    
      # => SHIT
    
    #if counter_below_cm == 0:                    
      # => SHIT
    #if counter_below_cm > 0:                     
      # => SHIT
    #if counter_below_cm == number_of_columns:    
      # => NO DIGITS  
    
    #if counter_above_mpmax == 0:                 
      # => SHIT
    #if counter_above_mpmax > 0:                  
      # => SHIT
    #if counter_above_mpmax == number_of_columns: 
      # => NO DIGITS  

    #if counter_below_mpmax == 0:                 
      # => NO DIGITS  
    #if counter_below_mpmax > 0:                  
      # => ALL DIGITS
    #if counter_below_mpmax == number_of_columns: 
      # => SHIT
    
    #if counter_above_mpmin == 0:                 
      # => NO DIGITS  
    #if counter_above_mpmin > 0:                  
      # => ALL DIGITS
    #if counter_above_mpmin == number_of_columns: 
      # => SHIT
      
    #if counter_below_mpmin == 0:                 
      # => SHIT
    #if counter_below_mpmin > 0:                  
      # => GOOD
    #if counter_below_mpmin == number_of_columns: 
      # => NO DIGITS       
      
    #if counter_above_cmin == 0:                  
      # => NO DIGITS  
    #if counter_above_cmin > 0:                   
      # => ALL DIGITS
    #if counter_above_cmin == number_of_columns:  
      # => SHIT      
      
    #if counter_below_cmin == 0:                  
      # => ALL DIGITS
    #if counter_below_cmin > 0:                   
      # => NO DIGITS  
    #if counter_below_cmin == number_of_columns:  
      # => NO DIGITS        
      
    #if counter_above_cmax == 0:                  
      # => ALL DIGITS
    #if counter_above_cmax > 0:                   
      # => NO DIGITS  
    #if counter_above_cmax == number_of_columns:  
      # => NO DIGITS       
      
    #if counter_below_cmax == 0:                  
      # => NO DIGITS  
    #if counter_below_cmax > 0:                   
      # => ALL DIGITS
    #if counter_below_cmax == number_of_columns:  
      # => SHIT      
      
    #if counter_between_cmin_cmax == 0:                   
      # => NO DIGITS  
    #if counter_between_cmin_cmax > 0:                    
      # => ALL DIGITS
    #if counter_between_cmin_cmax == number_of_columns:   
      # => SHIT      
      
    #if counter_between_mpmin_mpmax == 0:                 
      # => FEW DIGITS
    #if counter_between_mpmin_mpmax > 0:                  
      # => SHIT
    #if counter_between_mpmin_mpmax == number_of_columns: 
      # => SHIT      
      
    #if counter_between_cm_mpmax == 0:                    
      # => GOOD
    #if counter_between_cm_mpmax > 0:                     
      # => SHIT
    #if counter_between_cm_mpmax == number_of_columns:    
      # => FEW DIGITS      
      
    #if counter_between_cm_mpmin == 0:                    
      # => SHIT
    #if counter_between_cm_mpmin > 0:                     
      # => SHIT
    #if counter_between_cm_mpmin == number_of_columns:    
      # => NO DIGITS        
      
    #if counter_between_cm_cmin == 0:                     
      # => SHIT
    #if counter_between_cm_cmin > 0:                      
      # => GOOD
    #if counter_between_cm_cmin == number_of_columns:     
      # => NO DIGITS        
      
    #if counter_between_cm_cmax == 0:                     
      # => NO DIGITS  
    #if counter_between_cm_cmax > 0:                      
      # => ALL DIGITS
    #if counter_between_cm_cmax == number_of_columns:     
      # => SHIT      
      
    #if counter_between_mpmax_cmax == 0:                  
      # => SHIT
    #if counter_between_mpmax_cmax > 0:                   
      # => CHAOTIC
    #if counter_between_mpmax_cmax == number_of_columns:  
      # => NO DIGITS        
      
    #if counter_between_mpmin_cmin == 0:                  
      # => SHIT
    #if counter_between_mpmin_cmin > 0:                   
      # => GOOD
    #if counter_between_mpmin_cmin == number_of_columns:  
      # => NO DIGITS        
    
    # ------------------------------

    #if counter_above_cm > 0 and counter_below_cm > 0:
      # => SHIT
    #if counter_above_cm > 0 and counter_above_mpmax > 0:
      # => SHIT
    #if counter_above_cm > 0 and counter_below_mpmax > 0:
      # => ALL DIGITS
    #if counter_above_cm > 0 and counter_above_mpmin > 0:
      # => ALL DIGITS
    #if counter_above_cm > 0 and counter_below_mpmin > 0:
      # => GOOD
    #if counter_above_cm > 0 and counter_above_cmin > 0:
      # => ALL DIGITS
    #if counter_above_cm > 0 and counter_below_cmin > 0:
      # => NO DIGITS
    #if counter_above_cm > 0 and counter_above_cmax > 0:
      # => NO DIGITS
    #if counter_above_cm > 0 and counter_below_cmax > 0:
      # => ALL DIGITS
    #if counter_above_cm > 0 and counter_between_cmin_cmax > 0:
      # => ALL DIGITS
    #if counter_above_cm > 0 and counter_between_mpmin_mpmax > 0:
      # => SHIT
    #if counter_above_cm > 0 and counter_between_cm_mpmax > 0:
      # => SHIT
    #if counter_above_cm > 0 and counter_between_cm_mpmin > 0:
      # => SHIT
    #if counter_above_cm > 0 and counter_between_cm_cmin > 0:
      # => GOOD
    #if counter_above_cm > 0 and counter_between_cm_cmax > 0:
      # => ALL DIGITS
    #if counter_above_cm > 0 and counter_between_mpmax_cmax > 0:
      # => SHIT
    #if counter_above_cm > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD

    #if counter_below_cm > 0 and counter_above_mpmax > 0:
      # => CHAOTIC
    #if counter_below_cm > 0 and counter_below_mpmax > 0:
      # => GOOD
    #if counter_below_cm > 0 and counter_above_mpmin > 0:
      # => CHAOTIC
    #if counter_below_cm > 0 and counter_below_mpmin > 0:
      # => GOOD
    #if counter_below_cm > 0 and counter_above_cmin > 0:
      # => CHAOTIC
    #if counter_below_cm > 0 and counter_below_cmin > 0:
      # => NO DIGITS
    #if counter_below_cm > 0 and counter_above_cmax > 0:
      # => NO DIGITS
    #if counter_below_cm > 0 and counter_below_cmax > 0:
      # => CHAOTIC
    #if counter_below_cm > 0 and counter_between_cmin_cmax > 0:
      # => SHIT
    #if counter_below_cm > 0 and counter_between_mpmin_mpmax > 0:
      # => SHIT
    #if counter_below_cm > 0 and counter_between_cm_mpmax > 0:
      # => SHIT
    #if counter_below_cm > 0 and counter_between_cm_mpmin > 0:
      # => SHIT
    #if counter_below_cm > 0 and counter_between_cm_cmin > 0:
      # => CHAOTIC
    #if counter_below_cm > 0 and counter_between_cm_cmax > 0:
      # => CHAOTIC
    #if counter_below_cm > 0 and counter_between_mpmax_cmax > 0:
      # => GOOD
    #if counter_below_cm > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD

    #if counter_above_mpmax > 0 and counter_below_mpmax > 0:
      # => CHAOTIC
    #if counter_above_mpmax > 0 and counter_above_mpmin > 0:
      # => CHAOTIC
    #if counter_above_mpmax > 0 and counter_below_mpmin > 0:
      # => GOOD
    #if counter_above_mpmax > 0 and counter_above_cmin > 0:
      # => CHAOTIC
    #if counter_above_mpmax > 0 and counter_below_cmin > 0:
      # => NO DIGITS
    #if counter_above_mpmax > 0 and counter_above_cmax > 0:
      # => NO DIGITS
    #if counter_above_mpmax > 0 and counter_below_cmax > 0:
      # => GOOD
    #if counter_above_mpmax > 0 and counter_between_cmin_cmax > 0:
      # => GOOD
    #if counter_above_mpmax > 0 and counter_between_mpmin_mpmax > 0:
      # => SHIT
    #if counter_above_mpmax > 0 and counter_between_cm_mpmax > 0:
      # => SHIT
    #if counter_above_mpmax > 0 and counter_between_cm_mpmin > 0:
      # => SHIT
    #if counter_above_mpmax > 0 and counter_between_cm_cmin > 0:
      # => GOOD
    #if counter_above_mpmax > 0 and counter_between_cm_cmax > 0:
      # => CHAOTIC
    #if counter_above_mpmax > 0 and counter_between_mpmax_cmax > 0:
      # => CHAOTIC
    #if counter_above_mpmax > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD

    #if counter_below_mpmax > 0 and counter_above_mpmin > 0:
      # => ALL DIGITS
    #if counter_below_mpmax > 0 and counter_below_mpmin > 0:
      # => GOOD
    #if counter_below_mpmax > 0 and counter_above_cmin > 0:
      # => ALL DIGITS
    #if counter_below_mpmax > 0 and counter_below_cmin > 0:
      # => NO DIGITS
    #if counter_below_mpmax > 0 and counter_above_cmax > 0:
      # => NO DIGITS
    #if counter_below_mpmax > 0 and counter_below_cmax > 0:
      # => ALL DIGITS
    #if counter_below_mpmax > 0 and counter_between_cmin_cmax > 0:
      # => ALL DIGITS
    #if counter_below_mpmax > 0 and counter_between_mpmin_mpmax > 0:
      # => SHIT
    #if counter_below_mpmax > 0 and counter_between_cm_mpmax > 0:
      # => SHIT
    #if counter_below_mpmax > 0 and counter_between_cm_mpmin > 0:
      # => SHIT
    #if counter_below_mpmax > 0 and counter_between_cm_cmin > 0:
      # => CHAOTIC
    #if counter_below_mpmax > 0 and counter_between_cm_cmax > 0:
      # => ALL DIGITS
    #if counter_below_mpmax > 0 and counter_between_mpmax_cmax > 0:
      # => CHAOTIC
    #if counter_below_mpmax > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD

    #if counter_above_mpmin > 0 and counter_below_mpmin > 0:
      # => GOOD
    #if counter_above_mpmin > 0 and counter_above_cmin > 0:
      # => ALL DIGITS
    #if counter_above_mpmin > 0 and counter_below_cmin > 0:
      # => NO DIGITS
    #if counter_above_mpmin > 0 and counter_above_cmax > 0:
      # => NO DIGITS
    #if counter_above_mpmin > 0 and counter_below_cmax > 0:
      # => ALL DIGITS
    #if counter_above_mpmin > 0 and counter_between_cmin_cmax > 0:
      # => ALL DIGITS
    #if counter_above_mpmin > 0 and counter_between_mpmin_mpmax > 0:
      # => SHIT
    #if counter_above_mpmin > 0 and counter_between_cm_mpmax > 0:
      # => SHIT
    #if counter_above_mpmin > 0 and counter_between_cm_mpmin > 0:
      # => SHIT
    #if counter_above_mpmin > 0 and counter_between_cm_cmin > 0:
      # => GOOD
    #if counter_above_mpmin > 0 and counter_between_cm_cmax > 0:
      # => ALL DIGITS
    #if counter_above_mpmin > 0 and counter_between_mpmax_cmax > 0:
      # => CHAOTIC
    #if counter_above_mpmin > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD

    #if counter_below_mpmin > 0 and counter_above_cmin > 0:
      # => GOOD
    #if counter_below_mpmin > 0 and counter_below_cmin > 0:
      # => NO DIGITS
    #if counter_below_mpmin > 0 and counter_above_cmax > 0:
      # => NO DIGITS
    #if counter_below_mpmin > 0 and counter_below_cmax > 0:
      # => GOOD
    #if counter_below_mpmin > 0 and counter_between_cmin_cmax > 0:
      # => GOOD
    #if counter_below_mpmin > 0 and counter_between_mpmin_mpmax > 0:
      # => CHAOTIC
    #if counter_below_mpmin > 0 and counter_between_cm_mpmax > 0:
      # => SHIT
    #if counter_below_mpmin > 0 and counter_between_cm_mpmin > 0:
      # => GOOD
    #if counter_below_mpmin > 0 and counter_between_cm_cmin > 0:
      # => GOOD
    #if counter_below_mpmin > 0 and counter_between_cm_cmax > 0:
      # => GOOD
    #if counter_below_mpmin > 0 and counter_between_mpmax_cmax > 0:
      # => GOOD
    #if counter_below_mpmin > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD

    #if counter_above_cmin > 0 and counter_below_cmin > 0:
      # => NO DIGITS
    #if counter_above_cmin > 0 and counter_above_cmax > 0:
      # => NO DIGITS
    #if counter_above_cmin > 0 and counter_below_cmax > 0:
      # => ALL DIGITS
    #if counter_above_cmin > 0 and counter_between_cmin_cmax > 0:
      # => ALL DIGITS
    #if counter_above_cmin > 0 and counter_between_mpmin_mpmax > 0:
      # => SHIT
    #if counter_above_cmin > 0 and counter_between_cm_mpmax > 0:
      # => SHIT
    #if counter_above_cmin > 0 and counter_between_cm_mpmin > 0:
      # => SHIT
    #if counter_above_cmin > 0 and counter_between_cm_cmin > 0:
      # => GOOD
    #if counter_above_cmin > 0 and counter_between_cm_cmax > 0:
      # => ALL DIGITS
    #if counter_above_cmin > 0 and counter_between_mpmax_cmax > 0:
      # => CHAOTIC
    #if counter_above_cmin > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD

    #if counter_below_cmin > 0 and counter_above_cmax > 0:
      # => NO DIGITS
    #if counter_below_cmin > 0 and counter_below_cmax > 0:
      # => NO DIGITS
    #if counter_below_cmin > 0 and counter_between_cmin_cmax > 0:
      # => NO DIGITS
    #if counter_below_cmin > 0 and counter_between_mpmin_mpmax > 0:
      # => NO DIGITS
    #if counter_below_cmin > 0 and counter_between_cm_mpmax > 0:
      # => NO DIGITS
    #if counter_below_cmin > 0 and counter_between_cm_mpmin > 0:
      # => NO DIGITS
    #if counter_below_cmin > 0 and counter_between_cm_cmin > 0:
      # => NO DIGITS
    #if counter_below_cmin > 0 and counter_between_cm_cmax > 0:
      # => NO DIGITS
    #if counter_below_cmin > 0 and counter_between_mpmax_cmax > 0:
      # => NO DIGITS
    #if counter_below_cmin > 0 and counter_between_mpmin_cmin > 0:
      # => NO DIGITS

    #if counter_above_cmax > 0 and counter_below_cmax > 0:
      # => NO DIGITS
    #if counter_above_cmax > 0 and counter_between_cmin_cmax > 0:
      # => NO DIGITS
    #if counter_above_cmax > 0 and counter_between_mpmin_mpmax > 0:
      # => NO DIGITS
    #if counter_above_cmax > 0 and counter_between_cm_mpmax > 0:
      # => NO DIGITS
    #if counter_above_cmax > 0 and counter_between_cm_mpmin > 0:
      # => NO DIGITS
    #if counter_above_cmax > 0 and counter_between_cm_cmin > 0:
      # => NO DIGITS
    #if counter_above_cmax > 0 and counter_between_cm_cmax > 0:
      # => NO DIGITS
    #if counter_above_cmax > 0 and counter_between_mpmax_cmax > 0:
      # => NO DIGITS
    #if counter_above_cmax > 0 and counter_between_mpmin_cmin > 0:
      # => NO DIGITS

    #if counter_below_cmax > 0 and counter_between_cmin_cmax > 0:
      # => ALL DIGITS
    #if counter_below_cmax > 0 and counter_between_mpmin_mpmax > 0:
      # => SHIT
    #if counter_below_cmax > 0 and counter_between_cm_mpmax > 0:
      # => SHIT
    #if counter_below_cmax > 0 and counter_between_cm_mpmin > 0:
      # => SHIT
    #if counter_below_cmax > 0 and counter_between_cm_cmin > 0:
      # => CHAOTIC
    #if counter_below_cmax > 0 and counter_between_cm_cmax > 0:
      # => ALL DIGITS
    #if counter_below_cmax > 0 and counter_between_mpmax_cmax > 0:
      # => CHAOTIC
    #if counter_below_cmax > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD

    #if counter_between_cmin_cmax > 0 and counter_between_mpmin_mpmax > 0:
      # => SHIT
    #if counter_between_cmin_cmax > 0 and counter_between_cm_mpmax > 0:
      # => SHIT
    #if counter_between_cmin_cmax > 0 and counter_between_cm_mpmin > 0:
      # => SHIT
    #if counter_between_cmin_cmax > 0 and counter_between_cm_cmin > 0:
      # => GOOD
    #if counter_between_cmin_cmax > 0 and counter_between_cm_cmax > 0:
      # => ALL DIGITS
    #if counter_between_cmin_cmax > 0 and counter_between_mpmax_cmax > 0:
      # => CHAOTIC
    #if counter_between_cmin_cmax > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD

    #if counter_between_mpmin_mpmax > 0 and counter_between_cm_mpmax > 0:
      # => SHIT
    #if counter_between_mpmin_mpmax > 0 and counter_between_cm_mpmin > 0:
      # => SHIT
    #if counter_between_mpmin_mpmax > 0 and counter_between_cm_cmin > 0:
      # => SHIT
    #if counter_between_mpmin_mpmax > 0 and counter_between_cm_cmax > 0:
      # => SHIT
    #if counter_between_mpmin_mpmax > 0 and counter_between_mpmax_cmax > 0:
      # => SHIT
    #if counter_between_mpmin_mpmax > 0 and counter_between_mpmin_cmin > 0:
      # => CHAOTIC

    #if counter_between_cm_mpmax > 0 and counter_between_cm_mpmin > 0:
      # => SHIT
    #if counter_between_cm_mpmax > 0 and counter_between_cm_cmin > 0:
      # => SHIT
    #if counter_between_cm_mpmax > 0 and counter_between_cm_cmax > 0:
      # => SHIT
    #if counter_between_cm_mpmax > 0 and counter_between_mpmax_cmax > 0:
      # => SHIT
    #if counter_between_cm_mpmax > 0 and counter_between_mpmin_cmin > 0:
      # => SHIT

    #if counter_between_cm_mpmin > 0 and counter_between_cm_cmin > 0:
      # => SHIT
    #if counter_between_cm_mpmin > 0 and counter_between_cm_cmax > 0:
      # => SHIT
    #if counter_between_cm_mpmin > 0 and counter_between_mpmax_cmax > 0:
      # => SHIT
    #if counter_between_cm_mpmin > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD

    #if counter_between_cm_cmin > 0 and counter_between_cm_cmax > 0:
      # => GOOD
    #if counter_between_cm_cmin > 0 and counter_between_mpmax_cmax > 0:
      # => GOOD    
    #if counter_between_cm_cmin > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD    

    #if counter_between_cm_cmax > 0 and counter_between_mpmax_cmax > 0:
      # => GOOD    
    #if counter_between_cm_cmax > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD    

    #if counter_between_mpmax_cmax > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD    


    # ------------------------------     
    # ALL THE GOOD CONDITIONS EXTRACTED FROM ABOVE TESTS
    # We graded them to keep the bests.
    # Tier 1 Section contains the conditions we kept for our experimentations.
    # ------------------------------ 

    # TIER 3
    #if counter_between_cm_mpmax == 0:                    
      # => GOOD => GRADE: 8.3-9.5
    #if counter_above_cm > 0 and counter_below_mpmin > 0:
      # => GOOD => GRADE: 8-9.7
    #if counter_above_cm > 0 and counter_between_cm_cmin > 0:
      # => GOOD => GRADE: 7.8-9.4
    #if counter_below_cm > 0 and counter_below_mpmin > 0:
      # => GOOD => GRADE: 8-9.7
    #if counter_below_cm > 0 and counter_between_mpmax_cmax > 0:
      # => GOOD => GRADE: 8.2-9.4      
    #if counter_above_mpmax > 0 and counter_below_mpmin > 0:
      # => GOOD => GRADE: 8-9.8
    #if counter_above_mpmax > 0 and counter_between_cm_cmin > 0:
      # => GOOD => GRADE: 7.5-9.9
    #if counter_below_mpmin > 0 and counter_between_mpmax_cmax > 0:
      # => GOOD => GRADE: 8-10.4
    #if counter_above_cmin > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD => GRADE: 8-10.4
    #if counter_below_cmax > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD => GRADE: 8-9.8      
    #if counter_between_cm_cmin > 0 and counter_between_cm_cmax > 0:
      # => GOOD => GRADE: 7.7-10
    #if counter_between_cm_cmin > 0 and counter_between_mpmax_cmax > 0:
      # => GOOD => GRADE: 7.5-9.9    
    #if counter_between_cm_cmin > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD => GRADE: 8.1-9.8        
    #if counter_between_cm_cmin > 0:                      
      # => GOOD => GRADE: 7.8-10
    #if counter_between_mpmin_cmin > 0:                   
      # => GOOD => GRADE: 8-10.4

    # TIER 2
    #if counter_above_cm > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD => GRADE: 8.6-9.8
    #if counter_below_mpmax > 0 and counter_below_mpmin > 0:
      # => GOOD => GRADE: 8.6-9.8
    #if counter_below_mpmax > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD => GRADE: 8.6-9.8
    #if counter_above_mpmin > 0 and counter_below_mpmin > 0:
      # => GOOD => GRADE: 8.6-9.8
    #if counter_above_mpmin > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD => GRADE: 8.6-9.8 
    #if counter_below_mpmin > 0 and counter_below_cmax > 0:
      # => GOOD => GRADE: 8.6-9.8   
    #if counter_below_mpmin > 0 and counter_between_cm_cmax > 0:
      # => GOOD => GRADE: 8.6-9.8      
    #if counter_below_cm > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD => GRADE: 8.7-9.8
    #if counter_between_cm_cmax > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD => GRADE: 8.7-9.8 
    #if counter_above_mpmax > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD => GRADE: 8.7-9.9
    #if counter_between_mpmax_cmax > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD => GRADE: 8.7-9.9
    #if counter_between_cm_mpmin > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD => GRADE: 9.1-9.8 
    #if counter_below_mpmin > 0 and counter_between_cm_mpmin > 0:
      # => GOOD => GRADE: 9.2-9.8      
      
    # TIER 1
    # 1st conditions set referred as CS1 in experimentations.
    #if counter_below_mpmin > 0:
      # => GOOD => GRADE: 8.6-10.3
    # 2nd conditions set referred as CS2 in experimentations.
    #if counter_below_mpmin > 0 and counter_between_mpmin_cmin > 0:
      # => GOOD => GRADE: 8.6-10.4
    # 3rd conditions set referred as CS3 in experimentations.
    if counter_below_mpmin > 0 and counter_between_cm_cmin > 0:
      # => GOOD => GRADE: 8.7-10.4
      final_tuples.append(df[t])

  
  # Recreate df as a copy of final_tuples.
  df = []
  df = final_tuples
  df = DataFrame(df)
  df = df.values



# Resource Constraints Score
# Function to evaluate the mean constraints completion score of a specific resource using each constraint max_value. The percentage of completion is calculated toward these max_values.
def resource_constraints_score_reshape(resource_index): 
  scores = []
  for k in range(len(constraints)):
    # Constraint Score percentage
    score = df[resource_index][constraints[k][0]]/constraints[k][1]
    scores.append(score)
  mean_score = mean(scores) # mean constraints completion score
  return mean_score

# Reshape df by keeping the best constraints score resources.
def reshape_constraints_score_kings():
  global df
  final_tuples = []
  all_cs = []

  # For each resource, calculate its constraints score and save it in final_tuples.
  for k in range(len(df)):
    cs = resource_constraints_score_reshape(k)
    all_cs.append(cs)
    final_tuples.append({
                          'tuple':df[k],
                          'cs':cs
                        })
  
  # Calculate the mean_cs.
  mean_cs = mean(all_cs)
  
  # Reset df.
  df = []
  """
  # "KINGS"
  # Recreate df with tuples having a cs superior to mean_cs.
  for t in range(len(final_tuples)):
    if final_tuples[t]['cs'] > mean_cs:
      df.append(final_tuples[t]['tuple'])
  """
  # "RATS"
  # Recreate df with tuples having a cs inferior to mean_cs.
  for t in range(len(final_tuples)):
    if final_tuples[t]['cs'] < mean_cs:
      df.append(final_tuples[t]['tuple'])
  
  df = DataFrame(df)
  df = df.values
  




















