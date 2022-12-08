# --------------------------
# Constraints for Copy/Paste
# --------------------------

  constraints_variation_name = "V1_3C" # FOR FILE SAVE
  constraints = []
  if profile_number == "1":
    constraints.append([1, 12, 0.05])
    constraints.append([2, 11, 0.1])
    constraints.append([3, 10, 0.15])
  elif profile_number == "2":
    constraints.append([1, 13, 0.1])
    constraints.append([2, 12, 0.15])
    constraints.append([3, 11, 0.05])
  elif profile_number == "3":
    constraints.append([1, 14, 0.05])
    constraints.append([2, 13, 0.1])
    constraints.append([3, 12, 0.15])
  elif profile_number == "4": 
    constraints.append([1, 7, 0.15])
    constraints.append([2, 8, 0.05])
    constraints.append([3, 9, 0.1])
  elif profile_number == "5": 
    constraints.append([1, 6, 0.1])
    constraints.append([2, 7, 0.15])
    constraints.append([3, 8, 0.05])
  elif profile_number == "6": 
    constraints.append([1, 5, 0.15])
    constraints.append([2, 6, 0.05])
    constraints.append([3, 7, 0.1])

  if profile_number == "1":  
    criteria_min_or_max = ['max', 'max', 'max']
  elif profile_number == "2":
    criteria_min_or_max = ['max', 'min', 'max']
  elif profile_number == "3":
    criteria_min_or_max = ['min', 'max', 'min']
  elif profile_number == "4":  
    criteria_min_or_max = ['max', 'max', 'max']
  elif profile_number == "5":  
    criteria_min_or_max = ['min', 'min', 'min']
  elif profile_number == "6":  
    criteria_min_or_max = ['max', 'min', 'max']


  constraints_variation_name = "V1_2C" # FOR FILE SAVE
  constraints = []
  if profile_number == "1":
    constraints.append([1, 12, 0.05])
    #constraints.append([2, 11, 0.1])
    constraints.append([3, 10, 0.15])
  elif profile_number == "2":
    constraints.append([1, 13, 0.1])
    #constraints.append([2, 12, 0.15])
    constraints.append([3, 11, 0.05])
  elif profile_number == "3":
    constraints.append([1, 14, 0.05])
    #constraints.append([2, 13, 0.1])
    constraints.append([3, 12, 0.15])
  elif profile_number == "4": 
    constraints.append([1, 7, 0.15])
    #constraints.append([2, 8, 0.05])
    constraints.append([3, 9, 0.1])
  elif profile_number == "5": 
    constraints.append([1, 6, 0.1])
    #constraints.append([2, 7, 0.15])
    constraints.append([3, 8, 0.05])
  elif profile_number == "6": 
    constraints.append([1, 5, 0.15])
    #constraints.append([2, 6, 0.05])
    constraints.append([3, 7, 0.1])

  if profile_number == "1":  
    criteria_min_or_max = ['max', 'max', 'max']
  elif profile_number == "2":
    criteria_min_or_max = ['max', 'min', 'max']
  elif profile_number == "3":
    criteria_min_or_max = ['min', 'max', 'min']
  elif profile_number == "4":  
    criteria_min_or_max = ['max', 'max', 'max']
  elif profile_number == "5":  
    criteria_min_or_max = ['min', 'min', 'min']
  elif profile_number == "6":  
    criteria_min_or_max = ['max', 'min', 'max']


  constraints_variation_name = "V2_3C" # FOR FILE SAVE
  constraints = []
  if profile_number == "1":
    constraints.append([1, 9, 0.3])
    constraints.append([2, 8, 0.2])
    constraints.append([3, 7, 0.1])
  elif profile_number == "2":
    constraints.append([1, 11, 0.35])
    constraints.append([2, 10, 0.25])
    constraints.append([3, 9, 0.15])
  elif profile_number == "3":
    constraints.append([1, 10, 0.4])
    constraints.append([2, 9, 0.3])
    constraints.append([3, 8, 0.2])
  elif profile_number == "4": 
    constraints.append([1, 9, 0.05])
    constraints.append([2, 10, 0.05])
    constraints.append([3, 11, 0.05])
  elif profile_number == "5": 
    constraints.append([1, 10, 0.2])
    constraints.append([2, 10, 0.2])
    constraints.append([3, 10, 0.2])
  elif profile_number == "6": 
    constraints.append([1, 10, 0.4])
    constraints.append([2, 9, 0.3])
    constraints.append([3, 8, 0.2])

  if profile_number == "1":  
    criteria_min_or_max = ['min', 'min', 'max']
  elif profile_number == "2":
    criteria_min_or_max = ['max', 'max', 'min']
  elif profile_number == "3":
    criteria_min_or_max = ['max', 'min', 'max']
  elif profile_number == "4":  
    criteria_min_or_max = ['min', 'max', 'min']
  elif profile_number == "5":  
    criteria_min_or_max = ['max', 'max', 'max']
  elif profile_number == "6":  
    criteria_min_or_max = ['min', 'min', 'min']


  constraints_variation_name = "V2_2C" # FOR FILE SAVE
  constraints = []
  if profile_number == "1":
    constraints.append([1, 9, 0.3])
    #constraints.append([2, 8, 0.2])
    constraints.append([3, 7, 0.1])
  elif profile_number == "2":
    constraints.append([1, 11, 0.35])
    #constraints.append([2, 10, 0.25])
    constraints.append([3, 9, 0.15])
  elif profile_number == "3":
    constraints.append([1, 10, 0.4])
    #constraints.append([2, 9, 0.3])
    constraints.append([3, 8, 0.2])
  elif profile_number == "4": 
    constraints.append([1, 9, 0.05])
    #constraints.append([2, 10, 0.05])
    constraints.append([3, 11, 0.05])
  elif profile_number == "5": 
    constraints.append([1, 10, 0.2])
    #constraints.append([2, 10, 0.2])
    constraints.append([3, 10, 0.2])
  elif profile_number == "6": 
    constraints.append([1, 10, 0.4])
    #constraints.append([2, 9, 0.3])
    constraints.append([3, 8, 0.2])

  if profile_number == "1":  
    criteria_min_or_max = ['min', 'min', 'max']
  elif profile_number == "2":
    criteria_min_or_max = ['max', 'max', 'min']
  elif profile_number == "3":
    criteria_min_or_max = ['max', 'min', 'max']
  elif profile_number == "4":  
    criteria_min_or_max = ['min', 'max', 'min']
  elif profile_number == "5":  
    criteria_min_or_max = ['max', 'max', 'max']
  elif profile_number == "6":  
    criteria_min_or_max = ['min', 'min', 'min']


  constraints_variation_name = "V3_3C" # FOR FILE SAVE
  constraints = []
  if profile_number == "1":
    constraints.append([1, 7, 0])
    constraints.append([2, 6, 0])
    constraints.append([3, 5, 0])
  elif profile_number == "2":
    constraints.append([1, 9, 0])
    constraints.append([2, 8, 0])
    constraints.append([3, 7, 0])
  elif profile_number == "3":
    constraints.append([1, 8, 0])
    constraints.append([2, 7, 0])
    constraints.append([3, 6, 0])
  elif profile_number == "4": 
    constraints.append([1, 7, 0])
    constraints.append([2, 8, 0])
    constraints.append([3, 9, 0])
  elif profile_number == "5": 
    constraints.append([1, 8, 0])
    constraints.append([2, 8, 0])
    constraints.append([3, 8, 0])
  elif profile_number == "6": 
    constraints.append([1, 8, 0])
    constraints.append([2, 7, 0])
    constraints.append([3, 6, 0])

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