
# Algorithme pour la détection et la gestion des similarités (fusions)
# => TROP BASIQUE - PAS SCIENTIFIQUE

def detect_and_manage_similarity(ref_o):
  foreach comp_o de O
  {
    if comp_o.status == active
      bool_sim = detect_similarity(ref_o, comp_o)
      if bool_sim == true
        merge(ref_o, comp_o) # On merge ref_o dans comp_o
        ref_o.delete() # on supprime ref_o
        exit
  }


# Détection d’une similarité
# Deux objectifs ayant le même “Name” et le même “Stakeholder” sont similaires.
def detect_similarity(ref_o, comp_o):
	if ref_o.name == comp_o.name && ref_o.stakeholder == comp_o.stakeholder
		return true
	else
		return false


# Gestion d’une similarité par fusion
# On merge ref_o dans comp_o
def merge(ref_o, comp_o):
	# Priority => le plus important (weak < medium < strong)
  if(comp_o.priority == weak || comp_o.priority == medium) && (ref_o.priority == medium || ref_o.priority == strong)
    comp_o.priority = ref_o.priority

	# Targets : on ajoute tous les tuples Targets de ref_o dans comp_o
  # Il faut fusionner les tuples qui ont le même so_id.
  # Nous faisons le choix de conserver les valeurs les plus ambitieuses ou les plus englobantes lors des fusions des différents attributs.
  foreach t de ref_o.targets
    if t.so_id n’existe pas dans comp_o.targets
      comp_o.targets.add(t)
    else
      # On fusionne les tuples en opérant des moyennes.

        comp_o.targets(t.so_id).so_target_value = max(t.so_target_value ; comp_o.targets(t.so_id).so_target_value)

        comp_o.targets(t.so_id).cmin_so_target_value = min(t.cmin_so_target_value ; comp_o.targets(t.so_id).cmin_so_target_value)

        comp_o.targets(t.so_id).cmax_so_target_value = max(t.cmax_so_target_value ; comp_o.targets(t.so_id).cmax_so_target_value)

        comp_o.targets(t.so_id).timestamp_start = min(t.timestamp_start ; comp_o.targets(t.so_id).timestamp_start)
        comp_o.targets(t.so_id).cmin_start = min(t.cmin_start ; comp_o.targets(t.so_id).cmin_start)
        comp_o.targets(t.so_id).cmax_start = max(t.cmax_start ; comp_o.targets(t.so_id).cmax_start)

        comp_o.targets(t.so_id).timestamp_end = max(t.timestamp_end ; comp_o.targets(t.so_id).timestamp_end)
        comp_o.targets(t.so_id).cmin_end = min(t.cmin_end ; comp_o.targets(t.so_id).cmin_end)
        comp_o.targets(t.so_id).cmax_end = max(t.cmax_end ; comp_o.targets(t.so_id).cmax_end)

        comp_o.targets(t.so_id).duration = max(t.duration ; comp_o.targets(t.so_id).duration)
        comp_o.targets(t.so_id).cmin_duration = min(t.cmin_duration ; comp_o.targets(t.so_id).cmin_duration)
        comp_o.targets(t.so_id).cmax_duration = max(t.cmax_duration ; comp_o.targets(t.so_id).cmax_duration)

        # Construction de la séquence de reco mergée
        foreach action act de t.recommendation_sequence
          if act n’existe pas dans comp_o.targets(t.so_id).recommendation_sequence
            comp_o.targets(t.so_id).recommendation_sequence.add(act)
        # On trie comp_o.targets(t.so_id).recommendation_sequence selon les timestamp_start.
        comp_o.targets(t.so_id).recommendation_sequence.sort(timestamp_start)
		
	# On trie comp_o.targets selon l’attribut “order”.
	comp_o.targets.sort(order)


# Algorithme pour la détection et la gestion des conflits
# => TROP BASIQUE - PAS SCIENTIFIQUE

def detect_and_manage_conflict(ref_o):
	foreach comp_o de O
  {
    if comp_o.status == active
      bool_conflict = detect_conflict(ref_o, comp_o)
      if bool_conflict == true
        manage_conflict(ref_o, comp_o)
  }


# Détection d’un conflit
def detect_conflict(ref_o, comp_o):
  # Si ref_o et comp_o concernent le même stakeholder, on cherche des superpositions temporelles.
	if comp_o.stakeholder == ref_o.stakeholder
    # Recherche d’une superposition temporelle entre les tuples targets de ref_o et comp_o.
    # Dans Targets on utilise les attributs timestamp_start (start), timestamp_end (end) et duration.
    # overlap est un booléen prenant la valeur true si deux objectifs se chevauchent dans le temps et si les durations ne peuvent pas être mise bout à bout.
    foreach ref_o_t de ref_o.targets
      foreach comp_o_t de comp_o.targets
        # Chevauchement
        over = false
        if ref_o_t.start < comp_o_t.end || comp_o_t.start < ref_o_t.end
          over = true
        # Mise bout-à-bout des durations
        # On considère en première position le tuple ayant le “end - duration” le plus proche. Nous appelons ce tuple t_close et l’autre t_far.
        join_durations = true
        t_close = null
        t_far = null
        if(ref_o_t.end - ref_o_t.duration <= comp_o_t.end - comp_o_t.duration)
          t_close = ref_o_t
          t_far = comp_o_t
        else
          t_close = comp_o_t
          t_far = ref_o_t
        if t_close.end + t_far.duration > t_far.end
          join_durations = false
        # Y-a-t-il overlap ?
        if over == true && join_durations == false
          # Il y a overlap, il y a donc conflit entre ref_o et comp_o, il faut donc gérer ce conflit.
          return true
		return false


# Gestion d’un conflit

def manage_conflict(ref_o, comp_o):
  # Dans un premier temps, l’objectif qui a une priorité supérieure à l’autre verra ces tuples targets passer en premier. On rend ainsi l’autre “inactive”.
  if ref_o.priority > comp_o.priority
    comp_o.status = inactive
  else if ref_o.priority < comp_o.priority
    ref_o.status = inactive
  else 
    # les priority sont identiques, on considère les contraintes pour faire un compromis et ainsi conserver les deux objectifs. Si le compromis n’est pas possible, on rend ref_o “inactive” car on donne la priorité à comp_o déjà présent dans le système et potentiellement en cours.
    # Le compromis se fait avec les contraintes temporelles.
    # Est-il possible de concentrer les deux objectifs dans le temps ?
    # Nous utilisons les contraintes temporelles des tuples de targets : cmin_start, cmax_start, cmin_end, cmax_end, cmin_duration, cmax_duration
    # On récupère d’abord les valeurs min ou max des contraintes qui nous intéressent.
    
    ref_o_cmin_start = 0
    ref_o_cmax_end = 0
    ref_o_cmin_duration = 0
    foreach ref_o_t de ref_o.targets
      # cmin_start
      if ref_o_cmin_start == 0
          ref_o_cmin_start = ref_o_t.cmin_start
        else if ref_o_t.cmin_start < ref_o_cmin_start
          ref_o_cmin_start = ref_o_t.cmin_start
      # cmax_end
      if ref_o_cmax_end == 0
          ref_o_cmax_end = ref_o_t.cmax_end
        else if ref_o_t.cmax_end > ref_o_cmax_end
          ref_o_cmax_end = ref_o_t.cmax_end
      # cmin_duration
      if ref_o_cmin_duration == 0
          ref_o_cmin_duration = ref_o_t.cmin_duration
        else if ref_o_t.cmin_duration < ref_o_cmin_duration
          ref_o_cmin_duration = ref_o_t.cmin_duration
    
    comp_o_cmin_start = 0
    comp_o_cmax_end = 0
    comp_o_cmin_duration = 0
    foreach comp_o_t de comp_o.targets
      # cmin_start
      if comp_o_cmin_start == 0
          comp_o_cmin_start = comp_o_t.cmin_start
        else if comp_o_t.cmin_start < comp_o_cmin_start
          comp_o_cmin_start = comp_o_t.cmin_start
      # cmax_end
      if comp_o_cmax_end == 0
          comp_o_cmax_end = comp_o_t.cmax_end
        else if comp_o_t.cmax_end > comp_o_cmax_end
          comp_o_cmax_end = comp_o_t.cmax_end
      # cmin_duration
      if comp_o_cmin_duration == 0
          comp_o_cmin_duration = comp_o_t.cmin_duration
        else if comp_o_t.cmin_duration < comp_o_cmin_duration
          comp_o_cmin_duration = comp_o_t.cmin_duration

    time_concentration = false
    if second(ref_o_cmin_start ; comp_o_cmax_end) < second(ref_o_cmin_duration + comp_o_cmin_duration) || second(comp_o_cmin_start ; ref_o_cmax_end) < second(ref_o_cmin_duration + comp_o_cmin_duration)
      time_concentration = true

    # Si la concentration dans le temps est impossible, on fait le choix de garder comp_o car il est déjà présent dans le système. On supprime donc ref_o.
    if time_concentration == false
      ref_o.status = inactive
    # Si la concentration dans le temps est possible, on conserve les deux objectifs donc on ne touche à rien. En partant du principe que ref_o est enregistré par défaut avec status = active.


# Algorithme pour ordonner les objectifs concernant un acteur et sortir sa liste de recommandations


def sort_and_recommend(stakeholder):
  # Pour le stakeholder concerné, on récupère les tuples d’actions en état “todo” et “doing” des tuples targets en état “todo” et “doing” des objectifs en état “active” et dont les “targets.cmax_end” ne sont pas dépassés en appliquant les “targets.cmin_duration” à partir du moment présent.
  tuples_actions = {}
  foreach o de O
    if o.status == “active” && stakeholder = o.stakeholder
      foreach t de o.targets
        if (t.status = “todo” || t.status = “doing”) &&  (time(now) + t.cmin_duration <= t.cmax_end)
          foreach action de t.recommendation_sequence
            if (action.status = “todo” || action.status = “doing”)
              tuples_actions.add(action)

  # On trie ces tuples avec “timestamp_start”, “timestamp_end” et “duration” en utilisant un tri topologique.
  tuples_actions.TopologicalSort()

  # On propose au stakeholder concerné les actions dans l’ordre du tri effectué.
  foreach tuple de tuples_actions
    print tuple.action_verb .” “. tuple.resource


# TROUVER UN ALGO DE TRI TOPOLOGIQUE EN PYTHON ET LE METTRE CI-DESSOUS






# feed_with_neural_net()
def feed_with_neural_net(ref_o):

  # Suppression des fichiers temporaires
  if path.exists("extract_for_neural_net.csv"): 
    os.remove('extract_for_neural_net.csv')
  if path.exists("neural_net_model.h5"): 
    os.remove('neural_net_model.h5')  
     
  # Pour chaque sous-objectif de ref_o.
  for so_ref_o in ref_o.targets:
    if len(so_ref_o['recommendations']) == 0:
      
      # Préparation des données à passer en entrée du réseau de neurone.
      # Il s'agit d'un tableau .csv avec en dernière colonne le résultat à prédire.
      # Les colonnes d'avant peuvent varier suivant le contexte.
      # NOTA : La requête ci-dessous ainsi que le traitement qui s'ensuit n'est qu'un exemple.
      # Comprendre les éléments du select comme cela : Avec telle ressource et telle valeur cible de sous-objectif et telle durée d'investissement, l'apprenant à atteint tel résultat. Nous devons donc passer au réseau de neurones des tuples (resource, valeur_cible, durée) afin qu'il puisse nous donner le résultat final associé. Si le résultat obtenu est >= au résultat souhaité, la ressource du tuple sera recommandée.
      
      c.execute('''
                SELECT 
                  w_recommendations.resource as resource,
                  AVG(w_sub_objectives.so_target_value) as mean_so_target_value,
                  AVG(w_sub_objectives.duration) as mean_so_duration,
                  AVG(w_sub_objectives.so_current_value) as mean_so_current_value

                FROM w_objectives
                INNER JOIN w_sub_objectives
                ON w_objectives.id_objective = w_sub_objectives.id_objective
                INNER JOIN w_recommendations
                ON w_sub_objectives.id_sub_objective = w_recommendations.id_sub_objective
                
                WHERE w_sub_objectives.so_name = "''' + so_ref_o['so_name'] + '''"
                AND w_objectives.stakeholder_quality = "''' + ref_o.quality + '''"
                AND w_objectives.objective_status = "achieved"
                GROUP BY resource
                ''')
            
      conn.commit()

      df_raw_data = DataFrame(c.fetchall(), columns=['resource', 'mean_so_target_value', 'mean_so_duration', 'mean_so_current_value'])

      # Création du .csv
      # Il faut convertir tous les champs string en double/float/int afin que le réseau de neurones puisse travailler avec.
      with open('extract_for_neural_net.csv', 'a', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=';')
        
        filewriter.writerow(
                                [
                                  'resource',
                                  'mean_so_target_value',
                                  'mean_so_duration',
                                  'mean_so_current_value'
                                ]
                            )
        
        for index, row in df_raw_data.iterrows():
          resource = row['resource']
          mean_so_target_value = row['mean_so_target_value']
          mean_so_duration = row['mean_so_duration']
          mean_so_current_value = row['mean_so_current_value']

          filewriter.writerow(
                                [
                                  resource,
                                  mean_so_target_value,
                                  mean_so_duration,
                                  mean_so_current_value,
                                ]
                              )
          
      # On peut maintenant passer ces données au réseau de neurones
      
      # Load du fichier csv.
      df = pd.read_csv('extract_for_neural_net.csv', sep=";")
      #df = pd.read_csv('test.csv', sep=";")
      
      # Transformation des données en array.
      dataset = df.values
      print("")
      print(dataset)
      print("")
      
      # Création de notre matrice X (les colonnes auxquelles le réseau de neurones assignera des poids)
      # Everything before the comma refers to the rows of the array and everything after the comma refers to the columns of the arrays.
      # Since we’re not splitting up the rows, we put ‘:’ before the comma. This means to take all the rows in dataset and put it in X.

      # We want to extract out the first 2 column, and so the ‘0:2’ after the comma means take columns 0 to 1 excluded and put it in X (we don’t include column 2). Our columns start from index 0, so the first column is column 0.
      X = dataset[:,0:3]
      #print(X)
      
      # Création de notre vecteur résultat Y (la colonne à prédire)
      Y = dataset[:,3]
      #print(Y)
      
      # Enregistrement de la moyenne et de l'écart type de Y 
      # afin de pouvoir récupérer les vraies valeurs de prédiction,
      # c'est à dire les ids des resources à recommander.
      mean_y = mean(Y)
      #mean_y
      std_y = std(Y)
      #std_y
      
      # Normalisation de X
      # min-max scaler scales the dataset so that all the input features lie between 0 and 1 inclusive.
      min_max_scaler = preprocessing.MinMaxScaler()
      X_scale = min_max_scaler.fit_transform(X)
      #print(X_scale)
      
      # Création des jeux de données d'entraînement et de validation/test.
      # Les jeux de données de test représentent 30% (0.3) des données d'origine.
      X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
      
      # Redécoupage des jeux de données validation/test en jeux de données validation et jeux de données test. On coupe en deux, soit 50% (0.5).
      X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
      
      # Affiche les proportions de chaque jeu de données.
      #print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
      
      # Création du modèle.
      #Available activation functions : relu - sigmoid - softmax - softplus - softsign - tanh - selu - elu - exponential
      # Dense correspond à une couche de neurones fully-connected
      # 1ère couche : 32 neurones, input_shape = 2 car 2 colonnes en entrée.
      model = Sequential([    
                      Dense(16, activation='relu', input_shape=(3,)),    
                      Dense(32, activation='relu'),    
                      Dense(64, activation='relu'),
                      Dense(128, activation='relu'),
                      Dense(1, activation='sigmoid'),
                  ])
      model.add(keras.layers.BatchNormalization())
      model.add(keras.layers.Flatten())


      # Configuration du modèle.
      #Available optimizers : SGD - RMSprop - Adam - Adadelta - Adagrad - Adamax - Nadam - Ftrl
      model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
      
      # Model training
      # epochs = number of iterations
      hist = model.fit(X_train, Y_train, batch_size=64, epochs=100, validation_data=(X_val, Y_val))
      
      # Évaluation du modèle.
      model.evaluate(X_test, Y_test)[0] # Affichage de Loss
      model.evaluate(X_test, Y_test)[1] # Affichage de Accuracy

      # Save the model
      #model.save('neural_net_model.h5')
       
      # Load the model
      # NOTA : On pourra directement load le modèle au début s'il existe déjà.
      #model = load_model('neural_net_model.h5')

      # Génération des prédictions
      to_predict = min_max_scaler.fit_transform([
                                                  [546614, 60, 4],
                                                  [546686, 50, 15],
                                                  [3, 40, 5],
                                                  [999999, 99, 99]
                                                ])          
      print(to_predict)
      
      # Predict "Deprecated"
      #predictions = model.predict_classes(to_predict)
      
      # Predict for a model doing multi-class classification (ex : last layer with softmax activation).
      #predictions = argmax(model.predict(to_predict), axis=-1)
      
      # Predict for a model doing binary classification (ex : last layer with sigmoid activation).
      predictions = (model.predict(to_predict) > 0.5).astype("int32")
      print(predictions)
      
      # On retransforme les résultats de prédiction en valeur réelles c'est à dire en id de ressource. On utilise pour cela le mean_y et std_y calculés plus haut.
      predictions_real_values = predictions * std_y + mean_y
      print(predictions_real_values)

      # On affiche les prédictions.
      for i in range(len(to_predict)):
        print("")
        print('For input vector %s, predicted result is %d' 
                % (
                    to_predict[i], 
                    predictions_real_values[i]
                  )
              )   

      # Suppression du fichier extract_for_neural_net.csv
      os.remove('extract_for_neural_net.csv')




# Analyse des recommandations effectuées dans le cadre d’objectifs atteints et similaires à ref_o. 
# Les séquences de recommandations des objectifs atteints et similaires (sub_objectives proches) seront copiées au sein de ref_o.
def feed_with_achieved(ref_o):
  
  # Pour chaque sous-objectif de ref_o.
  for so_ref_o in ref_o.targets:
    # Récupération parmi les objectifs "achieved" du sous-objectif ayant le même nom que so_ref_o et dont l'écart de so_target_value avec celle de so_ref_o est le plus petit.
    c.execute('''
                SELECT 
                      id_sub_objective, 
                      ABS(''' + so_ref_o['so_target_value'] + ''' - so_target_value) as ecart
                FROM w_objectives
                INNER JOIN w_sub_objectives
                ON w_objectives.id_objective = w_sub_objectives.id_objective
                WHERE w_objectives.objective_status = "achieved"
                AND so_name = ''' + so_ref_o['so_name'] + '''
                ORDER BY ecart ASC
                LIMIT 1
              '''
            )
            
    conn.commit()

    df = DataFrame(c.fetchall(), columns=['id_sub_objective', 'ecart'])

    # On prend les recos de l'unique sous-objectif retourné pour les mettre dans so_ref_o.
    for index, row in df.iterrows():
      c.execute('''
                  SELECT action_verb, resource, timestamp_start, timestamp_end, duration, status
                  FROM w_recommendations
                  WHERE id_sub_objective = ''' + str(row['id_sub_objective']) + '''
                  GROUP BY action_verb, resource
                  ORDER BY timestamp_start
                '''
              )
              
      conn.commit()
      
      df2 = DataFrame(c.fetchall(), columns=['action_verb', 'resource', 'timestamp_start', 'timestamp_end', 'duration', 'status'])

      for index2, row2 in df2.iterrows():
        so_ref_o['recommendations'].append({
                                      'action_verb':row2['action_verb'],
                                      'resource':row2['resource'],
                                      'timestamp_start':row2['timestamp_start'],
                                      'timestamp_end':row2['timestamp_end'],
                                      'duration':row2['duration'],
                                      'status':row2['status']
                                    })
        # On fait l'INSERT dans la base.
        """
        c.execute('''
                    INSERT INTO w_recommendations (id_sub_objective, action_verb, resource, timestamp_start, timestamp_end, duration, status) 
                    VALUES( 
                            ''' + row['id_sub_objective'] + ''',
                            ''' + row2['action_verb'] + ''',
                            ''' + row2['resource'] + ''',
                            ''' + row2['timestamp_start'] + ''',
                            ''' + row2['timestamp_end'] + ''',
                            ''' + row2['duration'] + ''',
                            ''' + row2['status'] + '''
                          )
                  ''') 
        conn.commit()
        """

  # Affichage de ref_o.
  print("") 
  print("Affichage de ref_o :")
  print("Objective Name : " + ref_o.name)
  for t in ref_o.targets:
    print("Sub_objective Name : " + t['so_name'])
    for rs in t['recommendations']:
      print("Action : " + rs['action_verb'])
      print("Resource : " + rs['resource'])
  print("")





# Idem que feed_with_achieved() mais avec les objectifs non-atteints (actifs ou inactifs)
def feed_with_others(ref_o):

  # Pour chaque sous-objectif de ref_o.
  for so_ref_o in ref_o.targets:
    print("len(so_ref_o[recos] = " + str(len(so_ref_o['recommendations'])))
    if len(so_ref_o['recommendations']) == 0:
      # Récupération (parmi les objectifs "active" et "inactive") du sous-objectif ayant le même nom que so_ref_o et dont l'écart de so_target_value avec celle de so_ref_o est le plus petit.
      c.execute('''
                  SELECT 
                        id_sub_objective, 
                        ABS(''' + so_ref_o['so_target_value'] + ''' - so_target_value) as ecart
                  FROM w_objectives
                  INNER JOIN w_sub_objectives
                  ON w_objectives.id_objective = w_sub_objectives.id_objective
                  WHERE (w_objectives.objective_status = "active" 
                          OR w_objectives.objective_status = "inactive")
                  AND so_name = ''' + so_ref_o['so_name'] + '''
                  ORDER BY ecart ASC
                  LIMIT 1
                '''
              )
              
      conn.commit()

      df = DataFrame(c.fetchall(), columns=['id_sub_objective', 'ecart'])

      # On prend les recos de l'unique sous-objectif retourné pour les mettre dans so_ref_o.
      for index, row in df.iterrows():
        c.execute('''
                    SELECT action_verb, resource, timestamp_start, timestamp_end, duration, status
                    FROM w_recommendations
                    WHERE id_sub_objective = ''' + str(row['id_sub_objective']) + '''
                    GROUP BY action_verb, resource
                    ORDER BY timestamp_start
                  '''
                )
                
        conn.commit()
        
        df2 = DataFrame(c.fetchall(), columns=['action_verb', 'resource', 'timestamp_start', 'timestamp_end', 'duration', 'status'])

        for index2, row2 in df2.iterrows():
          so_ref_o['recommendations'].append({
                                        'action_verb':row2['action_verb'],
                                        'resource':row2['resource'],
                                        'timestamp_start':row2['timestamp_start'],
                                        'timestamp_end':row2['timestamp_end'],
                                        'duration':row2['duration'],
                                        'status':row2['status']
                                      })
          # On fait l'INSERT dans la base.
          """
          c.execute('''
                    INSERT INTO w_recommendations (id_sub_objective, action_verb, resource, timestamp_start, timestamp_end, duration, status) 
                    VALUES( 
                            ''' + row['id_sub_objective'] + ''',
                            ''' + row2['action_verb'] + ''',
                            ''' + row2['resource'] + ''',
                            ''' + row2['timestamp_start'] + ''',
                            ''' + row2['timestamp_end'] + ''',
                            ''' + row2['duration'] + ''',
                            ''' + row2['status'] + '''
                          )
                  ''')
        conn.commit()
        """

  # Affichage de ref_o.
  print("") 
  print("Affichage de ref_o :")
  print("Objective Name : " + ref_o.name)
  for t in ref_o.targets:
    print("Sub_objective Name : " + t['so_name'])
    for rs in t['recommendations']:
      print("Action : " + rs['action_verb'])
      print("Resource : " + rs['resource'])
  print("")




# Default Recommendation
# Les fonctions feed_x() précédentes n'ayant pas permi d'assigner des recommandations à l'ensemble des sous-objectifs de ref_o, on assigne aux sous-objectifs dépourvus de recommandations les recommandations par défaut. 
# Ces recommandations par défaut correspondent à la table w_default_recommendations qui tout comme w_recommendations est liée à w_sub_objectives par l'attribut id_sub_objective.
# Au moment de la création d'un objectif, l'acteur qui crée l'objectif devra donner les informations nécessaires pour remplir cette table.
def feed_with_defaults(ref_o):
  # Pour chaque sous-objectif de ref_o.
  for so_ref_o in ref_o.targets:
    print("len(so_ref_o[recos] = " + str(len(so_ref_o['recommendations'])))
    if len(so_ref_o['recommendations']) == 0:
      # On prend les recos par défaut.
      c.execute('''
                  SELECT action_verb, resource, timestamp_start, timestamp_end, duration, status
                  FROM w_default_recommendations
                  WHERE id_sub_objective = ''' + str(so_ref_o['id_sub_objective']) + '''
                  ORDER BY timestamp_start
                '''
              )
      
      conn.commit()
      
      df2 = DataFrame(c.fetchall(), columns=['action_verb', 'resource', 'timestamp_start', 'timestamp_end', 'duration', 'status'])

      for index2, row2 in df2.iterrows():
        so_ref_o['recommendations'].append({
                                      'action_verb':row2['action_verb'],
                                      'resource':row2['resource'],
                                      'timestamp_start':row2['timestamp_start'],
                                      'timestamp_end':row2['timestamp_end'],
                                      'duration':row2['duration'],
                                      'status':row2['status']
                                    })
        # On fait l'INSERT dans la base.
        """
        c.execute('''
                    INSERT INTO w_recommendations (id_sub_objective, action_verb, resource, timestamp_start, timestamp_end, duration, status) 
                    VALUES( 
                            ''' + so_ref_o['id_sub_objective'] + ''',
                            ''' + row2['action_verb'] + ''',
                            ''' + row2['resource'] + ''',
                            ''' + row2['timestamp_start'] + ''',
                            ''' + row2['timestamp_end'] + ''',
                            ''' + row2['duration'] + ''',
                            ''' + row2['status'] + '''
                          )
                  ''')
        conn.commit()
        """


















