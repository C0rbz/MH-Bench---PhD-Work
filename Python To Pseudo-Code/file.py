def increment\_all\_counters(t, c):
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