# Getting necessary library
import streamlit as st
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
import seaborn as sns
style.use("ggplot")
st.set_option('deprecation.showPyplotGlobalUse', False) # to handle depreciations
# Showing the data in raw form
st.title('Energy Efficiency of Buildings')
st.write("Find the reasearch paper here - https://github.com/hereiskunalverma/Random_Repository/blob/main/Energy%20Efficiency%20of%20Building/HLCL%20Reference%20Paper%201.pdf")


# Side Bar of WebApp
page = st.sidebar.selectbox("Choose a page", 
  ['Homepage', 
  'Visualizing All Correlations',
  'Model - RandomForestRegressor',
  "Feature Selection using Genetic Algorithm",
  'About Me'
])
DATA_URL = (r'excel for HLCL.csv')
@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    # data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

# Loading data for model
data = pd.read_csv(r'excel for HLCL.csv')
datavis = data # copying data for visualization

# Data Exploration as per research paper "Accurate quantitative estimation of energy performance of residential using statistical machine learning tools Athanasios Tsanasa,∗, Angeliki Xifarab"

# Normalizing the data as per research paper section-3.1

datavis = datavis.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
datavis_scaled = min_max_scaler.fit_transform(datavis)
datavis = pd.DataFrame(datavis_scaled)
if page=='Homepage':
  data_load_state = st.text('Loading data...')
  data = load_data(719)
  data_load_state.text("Data Loading... Done!")

  if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
  # Visualizing the entire dataset
  st.markdown("## **1. Visualizing the Entire Dataset** ")
  fig = plt.figure(figsize=(15,9))
  for a,b in enumerate(data.columns):
    ax = fig.add_subplot(4,3,a+1)
    sns.distplot(data[b])
  plt.tight_layout()
  st.pyplot(fig)

elif page=='Visualizing All Correlations':
  st.markdown( """ 
    ## **2. Visualizing the input variable with relation with output variable**""")
  st.markdown("## Plotting each input variable with each output variable")

  st.markdown("## - **For Heating Load**")
  x_labels = ['Relative compactness', 'Surface area', 'Wall area', 'Roof area',
         'Overall Height', 'Orientation', 'Glazing Area',
         'Glazing Area Distribution']
  data_new = datavis.drop([8,9], axis=1) # Only has input variables

  for i in data_new.columns:
    ax = datavis.plot(kind='scatter',x=i,y=8, color='red', title='Co-Relation with Heading Load')
    ax.set_xlabel(x_labels[i])
    ax.set_ylabel('Heating Load')
    # plt.show()
    st.pyplot()

  st.markdown("## - **For Cooling Load**")
  for i in data_new.columns:
    ax = datavis.plot(kind='scatter',x=i,y=9, color='red', title='Co-Relation with Cooling Load')
    ax.set_xlabel(x_labels[i])
    ax.set_ylabel('Cooling Load')
    st.pyplot()
    # plt.show()
  st.markdown("# **3. Correlation Between Input Variable And Output variable**")

  st.write("""## I. Spearman Co-Relation between each input variable and each output variable""")

  st.write("""
  The Spearman rank correlation coefficient can characterize
  general monotonic relationships and lies in the range
  −1 to 1, 
  where negative sign indicates inversely proportional and 
  positive sign indicates proportional relationship, whilst the magnitude
  denotes how strong this relationship""")
  ls = max(x_labels, key=len)
  st.markdown("## - For Heating Load")
  heat_load_sp = []
  for i in x_labels:
    rank_corr = stats.spearmanr(data[i], data['Heating Load'])
    heat_load_sp.append(rank_corr)
  st.write('1. If the correlation is closer to 1 or +ve, it is directly proportional to the output variable(heatload).')
  st.write('2. If the correaltion is closer to -1 or -ve, it is inversely proporational to the output variable(heatload).')
  st.write(' ')
  for a,b in enumerate(x_labels):
    res=b.rjust(len(ls))+" -->   "+str(heat_load_sp[a])
    st.write(str(a+1)+'. '+res)
  st.write()
  st.markdown("## - For Cooling Load")
  cool_load_sp = []
  for i in x_labels:
    rank_corr = stats.spearmanr(data[i], data['Cooling Load'])
    cool_load_sp.append(rank_corr)
  st.write('1. If the correlation is closer to 1 or +ve, it is directly proportional to the output variable(coolload).')
  st.write('2. If the correaltion is closer to -1 or -ve, it is inversely proporational to the output variable(coolload).')
  st.write(' ')
  for a,b in enumerate(x_labels):
    res = b.rjust(len(ls))+" -->   "+str(cool_load_sp[a])
    st.write(str(a+1)+'. '+res)

  st.write("""## II. Mutual Information(normalized)""")

  st.write("""
  1. MI can be used to quantify any arbitrary relationships between the input and output variables.
  2. Because MI is not upper bounded we normalize it to lie in the range [0 to 1]
  3. The larger the MI value, the stronger the association strength between the two variables 

  **Note** - The MI value may not matched with the MI values in the research paper but the difference is the same so 
  there is nothing to worry about still MI gives the same difference as in the research paper""")

  st.markdown("## - For heating load")
  ls = max(x_labels, key=len) # maximum string length for padding
  heat_load_mi = []
  for i in x_labels:
    mi = normalized_mutual_info_score(data[i],data['Heating Load'])
    heat_load_mi.append(mi)
  st.write('Heating Load MI: \n')
  for a,b in enumerate(x_labels):
    res = b.rjust(len(ls))+"  -->   "+str(heat_load_mi[a])
    st.write(str(a+1)+'. '+res)

  st.write(' ')

  st.write('## - For cooling head')
  cool_load_mi = []
  for i in x_labels:
    mi = normalized_mutual_info_score(data[i],data['Cooling Load'])
    cool_load_mi.append(mi)
  st.write('Cooling Load MI: \n')

  for a,b in enumerate(x_labels):
    
    res = b.rjust(len(ls))+"  -->  "+str(cool_load_mi[a])
    st.write(str(a+1)+'. '+res)
  st.markdown("# **4. Visualizing all Correlations** ")
  # For hot load
  coh = []
  ph = []
  for i in range(len(heat_load_sp)):
    cof,pv = heat_load_sp[i]
    coh.append(cof)
    ph.append(pv)
  coframe_heat_dict = {'InputVariable':x_labels, 'MutualInformaiton(normalized)':heat_load_mi, 'SpearmanrCorrelaitonCoefficients':coh, 'pvalue':["%.4f"%(i) for i in ph]}
  coframe_heat = pd.DataFrame(coframe_heat_dict)
  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)

  # For Cool Load
  coc = []
  pc = []
  for i in range(len(cool_load_sp)):
    cof,pv = cool_load_sp[i]
    coc.append(cof)
    pc.append(pv)
  coframe_cool_dict = {'InputVariable':x_labels, 'MutualInformaiton(normalized)':cool_load_mi, 'SpearmanrCorrelaiton Coefficients':coc, 'pvalue':["%.4f"%(i) for i in pc]}
  coframe_cool = pd.DataFrame(coframe_cool_dict)
  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)
  st.markdown("### Heating Load Correlation Dataframe")
  st.write(coframe_heat)
  st.markdown("### Cooling Load Correlation Visualization")
  st.write(coframe_cool)


elif page=='Model - RandomForestRegressor':
  # Splitting the data into input and output variables
  X = datavis.iloc[:,:-2].values
  y1 = datavis.iloc[:,-2].values
  y2 = datavis.iloc[:,-1].values
  # print(X.shape, y1.shape)
  # print(X.shape, y2.shape)

  # Splitting the dataset
  X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y1, y2, test_size=0.30, random_state=42)
  # print(X_train.shape)
  # print(X_test.shape)
  # print(y1_train.shape)
  # print(y1_test.shape)
  # print(y2_train.shape)
  # print(y2_test.shape)
  st.markdown("# **5. Model - Random Forest Regressor with 300 Trees**")
  # for heat load
  model_heatload = RandomForestRegressor(n_estimators=300, random_state=42)
  model_heatload.fit(X_train, y1_train)
  # for cool load
  model_coolload = RandomForestRegressor(n_estimators=300, random_state=42)
  model_coolload.fit(X_train, y2_train)
  st.write("Model is trained...")

  # Predictions on dataset
  y1_pred = model_heatload.predict(X_test) # Heat Load Predicion
  y2_pred = model_coolload.predict(X_test) # Cool Load Prediction
  st.write("Predictions done...")

  st.markdown("## R2 score Evaluation on Test Sets")
  score_heat = r2_score(y1_test, y1_pred)
  score_cool = r2_score(y2_test, y2_pred)
  st.write("R2 Score -")
  st.write(f"1. Heating Load - {score_heat}")
  st.write(f"2. Cooling Load - {score_cool}")
  st.markdown("""# **6. Explanation About the Plot**""")
  st.markdown("""
  1. From Heat Load prediction curve, we can state that the model predictions are much more relevant as it fits the data with a 45° as expected.
  2. From Cool Load prediction curve, we can state the same that model predictions are much more relevant as it fits the data with a 45° as expected.""")
  st.markdown("## Plotting the curve for predicted and actual values")
  st.markdown("### For Heat Load")
  plt.scatter(y1_pred, y1_test, c='blue', s=10)
  plt.plot(y1_pred, model_heatload.predict(X_test), color='red')
  plt.title('Heat Load')
  plt.xlabel('Predicted Values')
  plt.ylabel('Actual Values')
  st.pyplot()
  # plt.show()
  st.markdown("### For Cool Load")
  plt.scatter(y2_pred, y2_test, c='blue', s=10)
  plt.plot(y2_pred, model_coolload.predict(X_test), color='red')
  plt.title('Cool Load')
  plt.xlabel('Predicted Values')
  plt.ylabel('Actual Values')
  st.pyplot()
  # plt.show()



elif page=="About Me":
  me=r"myself.png"
  st.image(me,use_column_width='always')
  st.markdown("""
    By
  # Kunal Verma
  ## Indian Institute Of Information Technology, Kota
  ## Computer Science & Engineering
  ## 2020 - 2024
  # Visit me : https://hereiskunalverma.github.io/tlrc/index.html

  """)

# Genetic Algorithm
# <==============================================================================>
# this might takes a lot of time
# st.markdown("""# Feature Selection using Genetic Algorithm""")
# st.markdown("""
#   **Note** - This might takes a lot of time so these are computed results...
#   """)
# #defining various steps required for the genetic algorithm
# def initilization_of_population(size,n_feat):
#     population = []
#     for i in range(size):
#         chromosome = np.ones(n_feat,dtype=np.bool)
#         chromosome[:int(0.3*n_feat)]=False
#         np.random.shuffle(chromosome)
#         population.append(chromosome)
#     return population

# def fitness_score(population):
#     scores = []
#     for chromosome in population:
#         model_heatload.fit(X_train[:,chromosome],y2_train)
#         predictions = model_heatload.predict(X_test[:,chromosome])
#         scores.append(r2_score(y2_test,predictions))
#     scores, population = np.array(scores), np.array(population) 
#     inds = np.argsort(scores)
#     return list(scores[inds][::-1]), list(population[inds,:][::-1])

# def selection(pop_after_fit,n_parents):
#     population_nextgen = []
#     for i in range(n_parents):
#         population_nextgen.append(pop_after_fit[i])
#     return population_nextgen

# def crossover(pop_after_sel):
#     population_nextgen=pop_after_sel
#     for i in range(len(pop_after_sel)):
#         child=pop_after_sel[i]
#         child[3:7]=pop_after_sel[(i+1)%len(pop_after_sel)][3:7]
#         population_nextgen.append(child)
#     return population_nextgen

# def mutation(pop_after_cross,mutation_rate):
#     population_nextgen = []
#     for i in range(0,len(pop_after_cross)):
#         chromosome = pop_after_cross[i]
#         for j in range(len(chromosome)):
#             if random.random() < mutation_rate:
#                 chromosome[j]= not chromosome[j]
#         population_nextgen.append(chromosome)
#     #print(population_nextgen)
#     return population_nextgen

# def generations(size,n_feat,n_parents,mutation_rate,n_gen,X_train,
#                                    X_test, y_train, y_test):
#     best_chromo= []
#     best_score= []
#     population_nextgen=initilization_of_population(size,n_feat)
#     for i in range(n_gen):
#         scores, pop_after_fit = fitness_score(population_nextgen)
#         print(scores[:2])
#         pop_after_sel = selection(pop_after_fit,n_parents)
#         pop_after_cross = crossover(pop_after_sel)
#         population_nextgen = mutation(pop_after_cross,mutation_rate)
#         best_chromo.append(pop_after_fit[0])
#         best_score.append(scores[0])
#     return best_chromo,best_score

elif page=="Feature Selection using Genetic Algorithm":
  st.markdown("""# Feature Selection using Genetic Algorithm""")
  st.markdown("""
    **Note** - This might takes a lot of time so these are computed results...
    """)
  # st.markdown("## Running Genetic Algorithm... it might takes time")
  # chromo,score=generations(size=1000,n_feat=8,n_parents=100,mutation_rate=0.10,
  #                      n_gen=38,X_train=X_train,X_test=X_test,y_train=y1_train,y_test=y1_test)

  # model_heatload.fit(X_train[:,chromo[-1]],y2_train)
  # predictions = model_heatload.predict(X_test[:,chromo[-1]])
  st.write("R2 score after genetic algorithm is= ")
  st.markdown("""
[0.9707871290480345, 0.9707871290480345]

[0.9708876261318915, 0.9708876261318915]

[0.9708931054967246, 0.9708931054967246]

[0.9708931054967246, 0.9708931054967246]

[0.9708931054967246, 0.9708931054967246]

[0.9708931054967246, 0.9708931054967246]

[0.9708931054967246, 0.9708931054967246]

[0.9708931054967246, 0.9708931054967246]

[0.9708931054967246, 0.9708931054967246]

[0.9708931054967246, 0.9708931054967246]

[0.9708931054967246, 0.9708931054967246]

[0.9708931054967246, 0.9708931054967246]

[0.9708931054967246, 0.9708931054967246]

[0.9708931054967246, 0.9708931054967246]

[0.9708876261318915, 0.9708876261318915]

[0.9708931054967246, 0.9708931054967246]

[0.9708931054967246, 0.9708931054967246]

[0.9708931054967246, 0.9708931054967246]

[0.9708931054967246, 0.9708931054967246]

[0.9708931054967246, 0.9708931054967246]

[0.9708931054967246, 0.9708931054967246]

[0.9708931054967246, 0.9708931054967246]

[0.9708931054967246, 0.9708931054967246]

[0.9708931054967246, 0.9708931054967246]

[0.9708931054967246, 0.9708931054967246]

## R2 Score After running Genetic Algorithm is 0.9708931054234240
    """)