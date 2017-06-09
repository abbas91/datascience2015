#############################################
#
#
#    Probability Graphic Model 
#
# 
#############################################



# --- Installation
$ pip install pgmpy







'
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>
> "Bayesian Network Fundamental"
>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

-- Independence --
# -- Representing independencies using 'pgmpy'
# [1] Assertation (nodes)

from pgmpy.independencies import IndependenceAssertion

assertion1 = IndependenceAssertion('X', 'Y') 
" (X _|_ Y) "

assertion2 = IndependenceAssertion('X', 'Y', 'Z')
" (X _|_ Y|Z) "

assertion3 = IndependenceAssertion('X', ['Y','U','I'], ['Z','L','O'])
" (X _|_ Y,U,I|Z,L,O) "


# [2] A set of assertations (nodes)
from pgmpy.independencies import Independencies

independencies = Independencies() # empty object
independencies.get_assertations()
" [] "

independencies.add_assertations(assertion1, assertion2)
independencies.get_assertations()
" (X _|_ Y) " " (X _|_ Y|Z) "


independencies = Independencies(assertion1, assertion2)
independencies.get_assertations()
" (X _|_ Y) " " (X _|_ Y|Z) "

independencies = Independencies(['X', 'Y'],
	                            ['A', 'B', 'C'])
independencies.get_assertations()
" (X _|_ Y) " " (A _|_ B|C) "






-- Joint Probability --
# --- Representing joint probability distributions using pgmpy

from pgmpy.factors import JointProbabilityDistribution as Joint

distribution = Joint(['coin1', 'coin2'], # names of the variables
	                 [2, 2],             # number of states of each variable
	                 [0.25, 0.25, 0.25, 0.25]) # a list of probability values

distribution.check_independence('coin1', 'coin2')
" True "







-- Conditional Probability --
# --- Representing CPD in pgmpy

# Table CPD
from pgmpy.factor import TabularCPD 
quality = TabularCPD(variable='Quality', # name of variab;e
	                 variable_card=3,    # number of states (cardinality)
	                 values=[[0.3], [0.5], [0.2]]) # probability values

print(quality)
" ------------------------ "
" ['Quality', 0]   |   0.3 "
" ['Quality', 1]   |   0.5 "
" ['Quality', 2]   |   0.2 "
" ------------------------ "

quality.cardinality
" array([3]) "

quality.values
" array([0.3, 0.5, 0.2]) "

# Tree CPD
from pgmpy.factors import TreeCPD, Factor
tree_cpd = TreeCPD([
	       ('B', Factor(['A'], [2], [0.8, 0.2]), '0'),
	       ('B', 'C', '1'),
	       ('C', Factor(['A'], [2], [0.1, 0.9]), '0'),
	       ('C', 'D', '1'),
	       ('D', Factor(['A'], [2], [0.9, 0.1]), '0'),
	       ('D', Factor(['A'], [2], [0.4, 0.6]), '1')])

# Rule CPD
from pgmpy.factors import RuleCPD
rule = RuleCPD('A', {('A_0', 'B_0'): 0.8,
	                 ('A_1', 'B_0'): 0.2,
	                 ('A_0', 'B_1', 'C_0'): 0.4,
	                 ('A_1', 'B_1', 'C_0'): 0.6,
	                 ('A_0', 'B_1', 'C_1'): 0.9,
	                 ('A_1', 'B_1', 'C_1'): 0.1})





-- Bayesian Network Implementation --
# --- Representing a Bayesian Network in pgmpy
##################################################################################
#     ______________________                        ________________
#    |                      |                      |                |
#    |  Traffic Accident(A) |                      |  Heavy Rain(R) |
#    |______________________|                      |________________|
#           |                                              |
#           \                                              |
#            \             _________________               |
#             \           |                 |              |
#              \----------|  Traffic Jam(J) |---------------
#                         |_________________|
#                                  |     |
#                                  |     |
#     _____________________        |     |
#    |                     |       |     |
#    |  getting up late(G) |       |     |__________________________
#    |_____________________|       |                               |
#            |                     |                               |
#            |           __________|__________             ________|_________
#            |__________|                     |           |                  |
#                       |  Late for school(L) |           |  Long Queues(Q)  |
#                       |_____________________|           |__________________|
#
#
###################################################################################

from pgmpy.models import BayesianModel
model = BayesianModel()
# Add nodes to empty bayesian model
# ------------------------------------------------------ ( Traffic Accident -> traffic_jam )
# ------------------------------------------------------ ( Heavy Rain -> traffic_jam )
model.add_nodes_from(['rain', 'traffic_jam'])
model.add_edge('rain', 'traffic_jam')
# If add edge without adding node, node will be automatically added
"Example: "
model.add_edge('accident', 'traffic_jam')
model.nodes()
" ['accident', 'rain', 'traffic_jam'] "
model.edges()
" [('rain', 'traffic_jam'), ('accident', 'traffic_jam')}" # two edges showed
# each node has an associated CPD with it.
from pgmpy.factor import TabularCPD
cpd_rain = TabularCPD('rain', 2, [[0.4], [0.6]])
cpd_accident = TabularCPD('accident', 2, [[0.2], [0.8]])
cpd_traffic_jam = TabularCPD('traffic_jam', 2, 
	                         [[0.9, 0.6, 0.7, 0.1],
	                          [0.1, 0.4, 0.3, 0.9]],
	                          evdience=['rain', 'accident'],
	                          evidence_card=[2, 2])
# associate each CPD to model
model.add_cpds(cpd_rain, cpd_accident, cpd_traffic_jam)
model.get_cpds()
" [<TabularCPD representing P(rain: 2) at fsjidfsjdfaskdf>, "
" [<TabularCPD representing P(accident: 2) at fsxfgsdfgfsjdfaskdf>, "
" [<TabularCPD representing P(traffic_jam: 2 | rain:2, accident:2) at fsjidf234sjdfaskdf>, "



# Adding the remaining variables and their CPDs
# ------------------------------------------------------ ( traffic_jam -> long_queues )
model.add_node('long_queues')
model.add_edge('traffic_jam', 'long_queues')
cpd_long_queues = TabularCPD('long_queues', 2,
	                         [[0.9, 0.2],
	                          [0.1, 0.8]],
	                          evidence=['traffic_jam'],
	                          evidence_card=[2])
model.add_cpds(cpd_long_queues)


# ------------------------------------------------------ ( getting_up_late -> late_for_school )
# ------------------------------------------------------ ( traffic_jam -> late_for_school )
model.add_nodes_from(['getting_up_late',
	                  'late_for_school'])
model.add_edges_from([('getting_up_late', 'late_for_school'),
	                  ('traffic_jam', 'late_for_school')])

cpd_getting_up_late = TabularCPD('getting_up_late', 2,
	                             [[0.6], [0.4]])
cpd_late_for_school = TabularCPD('late_for_school', 2,
	                             [[0.9, 0.45, 0.8, 0.1],
	                              [0.1, 0.55, 0.3, 0.9]],
	                              evidence=['getting_up_late', 'traffic_jam'],
	                              evidence_card=[2, 2])

model.add_cpds(cpd_getting_up_late, cpd_late_for_school)
model.get_cpds()
" [<TabularCPD representing P(rain: 2)                                               at fsjidfsjdfaskdf>, "
" [<TabularCPD representing P(accident: 2)                                           at fsxfgsdfgfsjdfaskdf>, "
" [<TabularCPD representing P(traffic_jam: 2 | rain:2, accident:2)                   at fsjidf234sjdfaskdf>, "
" [<TabularCPD representing P(long_queues: 2 | traffic_jam:2)                        at fsjidf234sjdfaskdf>, "
" [<TabularCPD representing P(getting_up_late:2)                                     at fsjidf234sjdfaskdf>, "
" [<TabularCPD representing P(late_for_school: 2 | getting_up_late:2, traffic_jam:2) at fsjidf234sjdfaskdf>, "

# --- Check whether the model and all the associated CPDs are consistent
model.check_model()
" True "
# --- if any wrong/additional cpds
model.remove_cpds('wrong_cpds')
model.get_cpds()
" 'wrong cpd gone' "

-------------- P = P(A,R,J,G,L,Q) 
                 = P(A) P(R) P(J|A,R) P(Q|R) P(L|G,J)



# ----- Active Trail
model.is_active_trail('accident', 'rain')
" False "
model.is_active_trail('accident', 'rain',
	                   observed='traffic_jam')
" True "











'
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>
>   Markov Network Fundamental 
>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
########################################################
#
#
#                         _______
#                  -------|__A__|-------
#                  |                   |     
#                  |                   |
#               ___|___             ___|___
#               |__B__|             |__D__|
#                  |                   |
#                  |                   |
#                  |      _______      |
#                  -------|__C__|-------
#
#

########################################################

# Represent Factor
from pgmpy.factors import Factor

factor1 = Factor(['A', 'B'], [2, 2], [1000, 1, 5, 100]) # node names of the edge, states of each, weights value
print(factor1)
"  a          b          factor1(A, B)   "
"  A_0        B_0        1000            "
" .....................                  "


# Factor operation
-1-
# --- Marginalize over
factor1_marginalized = factor1.marginalize('B', inplace=False) # if inplace = T, original factor modified
factor1_marginalized.scope() # check scope of a factor
" ['A'] "
# --- Marginalize over multiple variables
factor2 = Factor(['A', 'B', 'C'],
	             [2, 2, 2],
	             np.arange(8))
factor2_marginalized = factor2.marginalize(['B', 'C'], inplace=False)
factor2_marginalized.scope() # check scope of a factor
" ['A'] "

-2-
# --- Reduction
factor1 = Factor(['A', 'B'], [2, 2], [1000, 1, 5, 100]) # node names of the edge, states of each, weights value
factor1_reduced = factor1.reduce(('B', 0), inplace=False) # if inplace = T, original factor modified
print(factor1_reduced) # listed A staes where B = 0
"  a       factor1(a)   "
"  A_0     1000         "
"  A_1     5            "
# --- Reduction over multiple variables
factor2 = Factor(['A', 'B', 'C'],
	             [2, 2, 2],
	             np.arange(8))
factor2_reduced = factor2.reduce([('B', 0), ('C', 1)], inplace=False)
print(factor2_reduced)
"  a      factor2(a)    "
"  A_0      ###         "
"  A_1      ###         "
factor2_reduced.scope()
" ['A'] "

-3-
# --- Factor production
factor1 = Factor(['A', 'B'], [2, 2], [1000, 1, 5, 100])
factor2 = Factor(['B', 'C'], [2, 3], [10, 30, 5, 120, 1500, 90])

factor_product = factor1 * factor2
" or "
factor_product = factor1.product(factor2)
factor_product.scope()
" ['A', 'B', 'C'] "
print(factor_product)
"  a    b    c      factor_product(a,b,c)  "
"  A_0  B_0  C_0    1000                   "
" .......................................  "

-4-
# --- factor Division
" f(a,b) / f(b) "
factor1 = Factor(['a','b'], [2,3], range(6))
factor2 = Factor(['b'], [3], range(3))

factor_divid = factor1 / factor2




# Represent Markov Network Model
from pgmpy.models import MarkovModel
model = MarkovModel([('A', 'B'), ('B', 'C')])
model.add_node('D')
model.add_edges_from([('C', 'D'), ('D', 'A')])
# --- Define factors to associate with model
from pgmpy.factors import Factor
factor_a_b = Factor(variables=['A', 'B'], cardibality=[2,2], values=[90. 100, 1, 10])
factor_b_c = Factor(variables=['B', 'C'], cardibality=[2,2], values=[10. 80, 70, 20])
factor_c_d = Factor(variables=['C', 'D'], cardibality=[2,2], values=[50. 120, 10, 10])
factor_d_a = Factor(variables=['D', 'A'], cardibality=[2,2], values=[20. 80, 50, 40])
# --- Add factors to model
model.add_factors(factor_a_b, factor_b_c, factor_c_d, factor_d_a)
model.get_factors()
" [<Factor representing phi(A:2, B:2)  at aidfoaiudn;aksndf> "
" [<Factor representing phi(B:2, C:2)  at aidfoaasdfsdfn;aksndf> "
" [<Factor representing phi(C:2, D:2)  at aidfoaiud234234sndf> "
" [<Factor representing phi(D:2, A:2)  at aawjdfblasjiudn;aksndf> "




# Cluster Graph -- "Cluster Graph"

########################################################
#
#              ________   _______   ________
#              |__f1__|---|__A__|---|__f3__|
#                  |                    |     
#                  |                    |
#               ___|___   ________   ___|___
#               |__B__|---|__f2__|---|__C__|
#             
#               
########################################################
# First import factor graph class from pgmpy.models
from pgmpy.models import FactorGraph
factor_graph = FactorGraph()

# Add variable nodes and factor nodes to model
factor_graph.add_nodes_from(['A','B','C','D','phi1','phi2','phi3'])

# Add edges between all nodes
factor_graph.add_edges_from([('A','phi1'), ('B','phi1'),
	                         ('B','phi2'), ('C','phi2'),
	                         ('C','phi3'), ('A','phi3')])


# Add factors into phi1, phi2, phi3
from pgmpy.factors import Factor
import numpy as np
phi1 = Factor(['A','B'], [2,2], np.random.rand(4))
phi2 = Factor(['A','B'], [2,2], np.random.rand(4))
phi3 = Factor(['A','B'], [2,2], np.random.rand(4))
factor_graph.add_factors(phi1, phi2, phi3)






# Cluster Graph -- Converting Markov model into a factor graph
from pgmpy.models import MarkovModel
mm = MarkovModel()
mm.add_nodes_from(['A','B','C'])
mm.add_edges_from([('A','B'),('B','C'),('C','A')])
mm.add_factors(phi1, phi2, phi3)

factor_graph_from_mm = mm.to_factor_graph()
" Factor nodes after conversions will be automatically added to form 'phi1_node1_node2...' "
factor_graph_from_mm.nodes()
" ['C', 'B', 'A', phi_A_B', 'phi_B_C', 'phi_A_C' "
factor_graph_from_mm.edges()
" [('phi_A_B', 'A'), ('phi_A_C', 'A') ... "





# Cluster Graph -- Converting Factor Graph to MarkovModel
########################################################
#
#                         _______  
#                         |__f__|
#                  __________|__________               
#                  |         |         |
#               ___|___   ___|___   ___|___
#               |__A__|---|__B__|---|__C__|
#             
#               
########################################################

phi = Factor(['A','B','C'], [2,2,2], np.random.rand(8))

factor_graph = FactorGraph()
factor_graph.add_nodes_from(['A','B','C','phi'])
factor_graph.add_edges_from([('A','phi'), ('B','phi'), ('C','phi')])

mm_from_factor_graph = factor_graph.to_markov_model()
mm_from_factor_graph.add_factor(phi) # After convert back to MM, all phi nodes lost, need to add back
mm_from_factor_graph.edges()
" [('B','A'), ('C','B'), ('C','A')] "







# Check local dependencies in Markov Network (D-separated)
from pgmpy.models import MarkovModel
mm = markovModel()
mm.add_nodes_from(['X1','X2','X3','X4','X5','X6','X7'])
mm.add_edges_from([('X1','X3'),('X1','X4'),('X2','X4') ... ])

mm.get_local_independencies()
" (X3 _|_ X5, X4, X7 | X6, X1) "
" (X4 _|_ X3, X5 | X6, X7, X1, X2) "
" (..............................) "





# Converting Bayesian Network into Markov Network
" In the process, Network lost some statistical indepedeneies, like V-structure,  "
" Change directed nodes to undirected "
" Connect node's parent nodes "


from pgmpy.models import BayesianModel
from pgmpy.factors import TabularCPD

# Create Bayesian networks
model = BayesianModel()
model.add_nodes_from(['Rain', 'TrafficJam'])
model.add_edges('Rain','TrafficJam')
model.add_edges('Accident', 'TrafficJam')
cpd_rain = TabularCPD('Rain', 2, [[0.4], [0.6]])
cpd_accident = TabularCPD('Accident', 2, [[0.2], [0.8]])
cpd_traffic_jam = TabularCPD(...)
..... "Same as chp 1 building Bayesian Model..."

# Convert to Markov Model
mm = model.to_markov_model()
mm.edges()
# Extract edges 'trangulate'
" [('TrafficJam', 'Accident'), "
"  (TrafficJam', 'LongQueues') "
"  ( .......................)  "
" ............................ "







# Converting Markov Network into Bayesian Network 
" add egdes between some nodes 'trangluate' "
" However,it leads to the loss of local independencies information "
from pgmpy.models import MarkovModel
from pgmpy.factors import Factor
model = MarkovModel()
model.add_nodes_from(....)
model.add_edges_from(....)

phi1_A_B = Factor(.......)
....... " Same as create Markov Network"

# Converting
bayesian_model = model.to_bayesian_model()
bayesian_model.edges()
" [('D','C'), (....) .....] " # Extract edges




** # Tranangulate a Graph to 'Chordal Graph'
from pgmpy.models import MarkovModel
from pgmpy.factors import Factor
import numpy as np
model = MarkovModel()
....... " Same as create Markov Model "

chordal_graph = model.triangulate()
chordal_graph.edges()
" [('C','D'), ('C','B') ..... " # Extract edges 

# There are 6 'Heuristic Algorithms for triangulation' can be choosed
Chordal_graph = model.triangulate(heuristic='H1') # H6 is defualt



























'
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>
>   Exact Inference - Asking Questions to Model
>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

-1- " Variable Elimination "
" More efficient than normalizing and marginalizing the probability distribution "
# make inference using Variable Elimination
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.factors import TabularCPD

# First create the model
resurant = BayesianModel([('location','cost'),
	                      ('quality','cost'),
	                      ('cost','no_of_people'),
	                      ('location','no_of_people')])
cpd_location = TabularCPD('location', 2, [[0.6,0.4]])
cpd_quality = TabularCPD('quality', 3, [[0.3,0.5,0.2]])
cpd_cost = TabularCPD('cost', 2, [[0.8,0.6,0.1,0.6,0.6,0.05], # 2 X 2 X 3 = 12 --> 6 each row X 2
	                              [0.2,0.1,0.9,0.4,0.4,0.95]],
	                              ['location','quality'], [2,3])
cpd_no_of_people = TabularCPD('no_of_people', 2, [[0.6,0.8,0.1,0.6], # 2 X 2 X 3 = 12 --> 6 each row X 2
	                                              [0.4,0.2,0.9,0.4]],
	                                              ['cost','location'], [2,2])

resurant.add_cpds(cpd_location, cpd_quality, cpd_cost, cpd_no_of_people)


# Creating the inference object of the model
resurant_inference = VariableElimination(resurant)

# Doing simple queries over one or multiple variables
resurant_inference.query(variables=['location'])

resurant_inference.query(variables=['location','no_of_people'])

resurant_inference.query(variables=['no_of_people'], evidence={'location':1, 'quality':1}) # If we have evidence

resurant_inference.query(variables=['no_of_people'], evidence={'location':1}, elimination_order=['quality', 'cost']) # can sepcify elimination sequence / otherwise system will choose automatically


-2- " Induced Graph "
" also defined as the undirected graph constructed by the unionof all the graphs formed in each step of variable elimination "
# Check induced graph
induced_graph = resurant_inference.induced_graph(['cost', 'location', 'no_of_people', 'quality'])
induced_graph.nodes()
" ['cost', 'location', 'no_of_people', 'quality'] "
induced_graph.edges()
" [('location', 'quality'), "
"  ('location', 'cost'), "
"  ('location .........  "
"  (...................)] "



-3- " Belief Propagation - Information passage during elimination "

** " Clique Tree - also called 'Junction Tree' "
" Undirected graph over a set of factors, where each factor represents a cluster of random variables and edges connect the clusters, "
" where scope has a nonempty intersection. "

##############################################
#
#     ___________   S1,2 = {C}  ________
#     |_A,_B,_C_|---------------|_C,_D_|
#         C1                       C2
#          |                     
#          |S1,3 = {B,C}        
#      ____|______             
#      |_E,_B,_C_|
#         C3                
#
#
###############################################

# We can define a clique tree
from pgmpy.models import JunctionTree
junction_tree = JunctionTree()

# each node in the junction tree is a cluster of random variables
# represented as a tuple
junction_tree.add_nodes_from([('A','B','C'),
	                          ('C','D')])
junction_tree.add_edges(('A', 'B', 'C'),
	                    ('C', 'D')) # Must has variable overlap between two cluster



# Create Clique Tree from a Model class
from pgmpy.models import BayesianModel, MarkovModel
from pgmpy.factors import TabularCPD, Factor

# Create a Bayesian Model
model = BayesianModel(....)

cpd_var1 = TabularCPD(....)
cpd_var2 = TabularCPD(....)
cpd_var3 = TabularCPD(....)
cpd_var4 = TabularCPD(....)
cpd_var5 = TabularCPD(....)

model.add_cpds(..........)

# Use 'to_junction' method on model class
junction_tree_bm = model.to_junction_tree()
type(junction_tree_bm)
" pgmpy.models.JunctionTree.JunctionTree "

junction_tree_bm.nodes()
" [('var1', 'var3), "
"  ('var1', 'var2', 'var5') "
"  (....................)] "

junction_tree_bm.edges()
" [('..............)] "



** " Message Passing - update belief of clusters "
" Using lauritzen-Spiegelhalter algorithm "
from pgmpy.models import BayesianModel
from pgmpy.factors import TabularCPD, Factor
from pgmpy.inference import BeliefPropagation

# create a bayesian model as we did before
model = BayesianModel(....)

cpd_var1 = TabularCPD(....)
cpd_var2 = TabularCPD(....)
cpd_var3 = TabularCPD(....)
cpd_var4 = TabularCPD(....)
cpd_var5 = TabularCPD(....)

model.add_cpds(..........)

# Apply propagation
belief_propagation = BeliefPropagation(model)

# To calibrate the clique tree, use calibrate() method
belief_propagation.calibrate()

# To get cluster (or clique) beliefs use the corresponding getters
belief_propagation.get_clique_beliefs()

# To get the sepset beliefs use the corresponding getters
belief_propagation.get_sepset_beliefs()

>> # Query variables not in the same cluster
belief_propagation.query(variables=['no_of_people'], evidence={'location':1, 'quality':1})

>> # Can apply MAP_Query - next
belief_propagation.map_query(variables=['no_of_people'], evidence={'location':1, 'quality':1})
" {'no_of_people': 0} "







-4- " MAP - Maximize A Posterior Probability "
" Given the current states to find out the maximized predicted var state - prediction "
" Different from Query which only care find out the distribution of target var over all states "


** " MAP using Variable Elimination "
" Using factor maximization "
from pgmpy.models import BayesianModel
from pgmpy.factors import TabularCPD
from pgmpy.inference import VariableElimination

# Create a Bayesian model
model = BayesianModel(....)

cpd_var1 = TabularCPD(....)
cpd_var2 = TabularCPD(....)
cpd_var3 = TabularCPD(....)
cpd_var4 = TabularCPD(....)
cpd_var5 = TabularCPD(....)

model.add_cpds(..........)

# Calculating the max marginals
model_inference = VariableElimination(model)
model_inference.max_marginal(variables=['late_for_school'])
" 0.571425378538 "
model_inference.max_marginal(variables=['late_for_school', 'traffic_jam'])
" 0.405745234235 "
model_inference.max_marginal(variables=['late_for_school'], evidence={'traffic_jam':1}) # With evidence
" 0.571425378538 "
model_inference.max_marginal(variables=['no_of_people'], evidence={'location':1}, elimination_order=['quality', 'cost']) # can sepcify elimination sequence / otherwise system will choose automatically
" 0.623481338534 "






** " MAP using belief propagation "
" When sum-product variable elimination is compuattional intractable "
from pgmpy.models import BayesianModel
from pgmpy.factors import TabularCPD
from pgmpy.inference import VariableElimination

# Create a Bayesian model
model = BayesianModel(....)

cpd_var1 = TabularCPD(....)
cpd_var2 = TabularCPD(....)
cpd_var3 = TabularCPD(....)
cpd_var4 = TabularCPD(....)
cpd_var5 = TabularCPD(....)

model.add_cpds(..........)

# Calculating the max marginals
model_inference = VariableElimination(model)
model_inference.map_query(variables=['late_for_school'])
" {'late_for_school': 0} "
model_inference.map_query(variables=['late_for_school', 'accident'])
" {'accidnet': 1, late_for_school': 0} "
model_inference.map_query(variables=['late_for_school'], evidence={'accident': 1})
" {'late_for_school': 0} "
model_inference.map_query(variables=['no_of_people'], evidence={'location':1}, elimination_order=['quality', 
                                                                                                  'cost',
                                                                                                  'location']) # can sepcify elimination sequence / otherwise system will choose automatically






-5- " Using Model for Prediction - Example "
# First, let's import and create dataset
import numpy as np
from pgmpy.models import BayesianModel

data = np.random.randint(low=0, high=2, size=(1000, 4))
data
" ([[0, 1, 0, 0], "
"   [1, 1, 1, 0], "
"   [..........], "
"    ...          "
"   [1, 0, 0, 0]])"

import pandas as pd
data = pd.DataFrame(data, columns=['cost','quality','location','no_of_people'])
data
"   cost   quality   location   no_of_people "
" 0    0         1          0              0 "
" 1    1         1          1              0 "
" .......................................... "
" 998  1         0          0              0 " 


train = data[:750]
test = data[:750].drop('no_of_people', axis=1) # Drop target var

# Create Bayesian Model structure
resurant_model = bayesian([('location','cost'),
	                       ('quality','cost'),
	                       ('location','no_of_people'),
	                       ('cost','no_of_people')])
resurant_model.fit(train)

# Fit to compute the cpd of all the variables from the training data
resurant_model.get_cpds()
" [<pgmpy.factors.CPD.tabluarCPD at fsjidfsjdfaskdf>, "
"  <pgmpy.factors.CPD.tabluarCPD at fsjidfsefgaskdf>, "
"  <pgmpy.factors.CPD.tabluarCPD at fsjidfsjdfattbt>, "
"  <pgmpy.factors.CPD.tabluarCPD at fsjdfgdwwergdfg>, "
" ..................................................] "

# Now for predicting the values of 'no_of_people' using this model
resurant_model.predict(test).values.ravel()
" array([1, 1, 1, 1, 0, 0, 1, ..............]) " # predicted 'no_of_people' on test set






























'
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>
>         Approximate Inference
>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>




>>>>>>>> " Propagation Based Approximation "

-1- " Cluster Graph Belief Propagation "
from pgmpy.models import BayesianModel
from pgmpy.inference import ClusterBeliefPropagation as CBP
from pgmpy.factors import TabularCPD

# Create a Bayesian model
model = BayesianModel(....)

cpd_var1 = TabularCPD(....)
cpd_var2 = TabularCPD(....)
cpd_var3 = TabularCPD(....)
cpd_var4 = TabularCPD(....)
cpd_var5 = TabularCPD(....)

model.add_cpds(..........)

# Apply Cluster Graph Belief Propagation
cluster_inference = CBP(model)
cluster_inference.query(variables=['cost'])
cluster_inference.query(variables=['cost'], evidence={'no_of_people': 1, 'quality': 0})





>>>>>>>> " Propagation Based Approximation with Message Approximation "

-2- " Cluster Graph Belief Propagation on [MAP] "
from pgmpy.models import BayesianModel
from pgmpy.inference import ClusterBeliefPropagation as CBP
from pgmpy.factors import TabularCPD

# Create a Bayesian model
model = BayesianModel(....)

cpd_var1 = TabularCPD(....)
cpd_var2 = TabularCPD(....)
cpd_var3 = TabularCPD(....)
cpd_var4 = TabularCPD(....)
cpd_var5 = TabularCPD(....)

model.add_cpds(..........)

# Apply Cluster Graph Belief Propagation
cluster_inference = CBP(model)
cluster_inference.map_query(variables=['cost'])
cluster_inference.map_query(variables=['cost'], evidence={'no_of_people': 1, 'quality': 0})





>>>>>>>> " Sampling Based Approximation "


-1- " Forward Sampling "



-2- " Importance Sampling "








>>>>>>>> " MCMC Based Approximation "


-1- " Gibbs Sampling "

























'
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>
>   Model Learning - Parameter Estimation in Bayesian Networks
>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

" Goal of Learning: Find a distribution M* very close to the underlaying real world distribution P "

" [1] - Density Estimation = Inference a distribution from M* "
" [2] - Predicting the specific probability values = To learn the comlpete underlaying probability distribution P "
" [3] - Knowledge Discovery = predicting the correct network structure "



>>>>>>>>>>>>>>> [1] " Learning Parameters "


-1- " Using MLE "
# Learning parameters in Bayesian Network
import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator


# ----------------------------------- Example 1
# Generating some random data
raw_data = np.random.randint(low=0, high=2, size=(100,2))
print(raw_data)
" array([[1, 1], "
"        [1, 1], "
"        [0, 1], "
"        ......  "
"        [0, 0]])"

data = pd.DataFrame(raw_data, columns=['X', 'Y'])
print(data) # Two coins toss result
"   X  Y "
"0  1  1 "
" ......."
"98 0  0 "

# Two coin model assuming that they are dependent
coin_model = BayesianModel([('X', 'Y')])
coin_model.fit(data, estimator=MaximumLikelihoodEstimator)
cpd_x = coin_model.get_cpds('X')
print(cpd_x)

"   X_0  |  0.46  "
"   X_1  |  0.54  "


# ------------------------------------ Example 2
raw_data = np.random.randint(low=0, high=2, size=(1000, 6))
data = pd.DataFrame(raw_data, columns=['A','R','J','G','L','Q'])

student_model = BayesianModel([('A','J'), ('R','J'), ('J','Q'), ('J','L'), ('G','L')])

student_model.fit(data, estimator=MaximumLikelihoodEstimator)
student_model.get_cpds()
" <TabluarCPD representing P(A: 2) at ejbdfouaeboidjfnaeif>, "
" <TabluarCPD representing P(R: 2) at ejbdfouaeboidjfnaeif>, "
" .......................................................... "




>>>>>>>>>>>>>> [2] " Learning Structure "


-1- " Using Bayesian Score "
import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator

# ----------------------------------- Example 1
# Generating some random data
raw_data = np.random.randint(low=0, high=2, size=(1000,2))
data = pd.DataFrame(raw_data, columns=['X', 'Y'])
"   X  Y "
"0  1  1 "
" ......."
"98 0  0 "

# fit the data with empty model, let it learn
coin_model = BayesianModel()
coin_model.fit(data, estimators=BayesianEstimator)
coin_model.get_cpds()
" <TabluarCPD representing P(X: 2) at ejbdfouaeboidjfnaeif>, "
" <TabluarCPD representing P(Y: 2) at ejbdfouaeboidjfnaeif>, "
coin_model.nodes()
" ['X', 'Y'] "
coin_model.edges() # data generated from random sets - no edges
" [] "



# ----------------------------------- Example 2
# Generating some random data
raw_data = np.random.randint(low=0, high=2, size=(1000, 6))
data = pd.DataFrame(raw_data, columns=['A','R','J','G','L','Q'])

student_model = BayesianModel()
student_model.fit(data, estimators=BayesianEstimator)
coin_model.get_cpds()
" <TabluarCPD representing P(A: 2) at ejbdfouaeboidjfnaeif>, "
" <TabluarCPD representing P(R: 2) at ejbdfouaeboidjfnaeif>, "
" .......................................................... "
coin_model.nodes()
" ['A','R','J','G','L','Q'] "
coin_model.edges() # data generated from random sets - no edges
" [] "


























'
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>
>   Model Learning - Parameter Estimation in Markov Metwork
>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

" Goal of Learning: Find a distribution M* very close to the underlaying real world distribution P "

" [1] - Density Estimation = Inference a distribution from M* "
" [2] - Predicting the specific probability values = To learn the comlpete underlaying probability distribution P "
" [3] - Knowledge Discovery = predicting the correct network structure "



>>>>>>>>>>>>>>> [1] " Learning Parameters "

-1- "MLE - exact"

import numpy as np
import pandas as pd
from pgmpy.models import MarkovModel
from pgmpy.estimators import MaximumLikelihoodEstimator

# Generating some data
raw_data = np.random.randint(low=0, high=2, size=(100,2))
print(raw_data)
" array([[1, 1], "
"        [1, 1], "
"        [0, 1], "
"        ......  "
"        [0, 0]])"

data = pd.DataFrame(raw_data, columns=['A', 'B'])
print(data) # Two coins toss result
"   X  Y "
"0  1  1 "
" ......."
"98 0  0 "

# Markov Model 
markov_model = MarkovModel([('A','B')])
markov_model.fit(data, estimator=MaximumLikelihoodEstimator)

factors = markov_model.get_factors()
print(factors[0])
"    A      B      phi(A,B)  "
"    A_0    B_0    0.100     "
"    A_0    B_1    0.200     "
" .......................... "




-2- "Approximate Inference - <Belief Propagation and pseudo-moment matching> "

import numpy as np
import pandas as pd
from pgmpy.models import MarkovModel
from pgmpy.estimators import PseudoMomentMatchingEstimator

# Generating some data
raw_data = np.random.randint(low=0, high=2, size=(100,4))
print(raw_data)
" array([[1, 1, 1, 1], "
"        [1, 1, 0, 1], "
"        [0, 1, 1, 0], "
"        ......  "
"        [0, 0, 0, 0]])"

data = pd.DataFrame(raw_data, columns=['A', 'B', 'C', 'D'])
print(data) # Two coins toss result
"   A  B  C  D "
"0  1  1  1  1 "
" ............."
"98 0  0  0  0 "


# Markov Model 
markov_model = MarkovModel([('A','B'), ('B','C'), ('C','D'), ('D','A')])
markov_model.fit(data, estimator=PseudoMomentMatchingEstimator)

factors = markov_model.get_factors()
factors 
" <Factor representing Phi(A: 2, B:2) at ejbdfouaeboidjfnaeif>, "
" <Factor representing Phi(B: 2, C:2) at ejbdfouaeboidjfnaeif>, "
" ............................................................. "






>>>>>>>>>>>>>> [2] " Learning Structure "

-1- " MLE - exact learning for sturcture "
import numpy as np
import pandas as pd
from pgmpy.models import MarkovModel
from pgmpy.estimators import MaximumLikelihoodEstimator

# Generating some data
raw_data = np.random.randint(low=0, high=2, size=(1000,2))
print(raw_data)
" array([[1, 1], "
"        [1, 1], "
"        [0, 1], "
"        ......  "
"        [0, 0]])"

data = pd.DataFrame(raw_data, columns=['A', 'B'])
print(data) # Two coins toss result
"   X  Y "
"0  1  1 "
" ......."
"98 0  0 "

# Markov Model 
markov_model = MarkovModel()
markov_model.fit(data, estimator=MaximumLikelihoodEstimator)

markov_model.get_factors()
" <Factor representing Phi(A: 2, B:2) at ejbdfouaeboidjfnaeif>, "
markov_model.nodes()
" ['A', 'B'] "
markov_model.edges()
" [('A','B')] "




-2- " Bayesian Score for learning structure "
import numpy as np
import pandas as pd
from pgmpy.models import MarkovModel
from pgmpy.estimators import BayesianEstimator

# Generating some data
raw_data = np.random.randint(low=0, high=2, size=(1000,2))
print(raw_data)
" array([[1, 1], "
"        [1, 1], "
"        [0, 1], "
"        ......  "
"        [0, 0]])"

data = pd.DataFrame(raw_data, columns=['A', 'B'])
print(data) # Two coins toss result
"   X  Y "
"0  1  1 "
" ......."
"98 0  0 "

# Markov Model 
markov_model = MarkovModel()
markov_model.fit(data, estimator=BayesianEstimator)

markov_model.get_factors()
" <Factor representing Phi(A: 2, B:2) at ejbdfouaeboidjfnaeif>, "
markov_model.nodes()
" ['A', 'B'] "
markov_model.edges()
" [('A','B')] "













'
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>
>   Specialized Models
>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

-0- " Naive Bayes Model " >>> "Classift documents"

>1 " Multivariate Bernoulli Naive Bayes Model " > "Better at small vocabulary size"
" A document represents by a vector of features with each indicates whether a word present or not in the document "

from sklearn.feature_extraction.text import CountVectorizer
# Example 1
vectorizer = CountVectorizer(min_df=1) # ignore any word frequency = 1 / counts

corpus = ['This is the first document.',
          'This is the second second document',
          'And the thrid one.',
          'Is this the first document?']

# fit_transform method basically learn the vocabulary dictionary and return term-document matrix
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names())
" ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'thrid', 'this'] "

print(X.toarray())
" [0, 1, 0, 1, 1, 2, 0, 0, 1], "
" [ ....................... ],"
" [.........................],"





# Example 2
vectorizer_binary = CountVectorizer(min_df=1, binary=True) # ignore any word frequency = 1 / T/F

x_binary = vectorizer_binary.fit_transform(corpus)

print(x_binary.toarray())
" [0, 1, 0, 1, 1, 1, 0, 0, 1], "
" [ ....................... ],"
" [.........................],"




# Example 3
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ['This is the first document.',
          'This is the second second document',
          'And the thrid one.',
          'Is this the first document?']

vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
" ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'thrid', 'this'] "
print(X.toarray())
" [0.56345, 0.123426, 0.01264, 0.00875, 0., 0., 0.1487, 0., 0.006591], "
" [ ............................................................... ], "
" [.................................................................], "




#------------ Start Model Example
from sklearn.datasets import fetch_20newgroups # 20 news groups datasets
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics


categories = ['alt.atheism','talk.religion.misc','comp.graphics','sci.space']

# Loading train data
data_train = fetch_20newgroups(subset='train',
	                           categories=categories,
	                           shuffle=True,
	                           random_state=42)

data_test = fetch_20newgroups(subset='test',
	                           categories=categories,
	                           shuffle=True,
	                           random_state=42)
y_train, y_test = data_train.target, data_test.target


# Vectorizing data
feature_extractor_type = "hashed"

if feature_extractor_type == "hashed":
	vectorizer = HashingVectorizer(stop_words='english')
elif feature_extraction_type == "count":
	vectorizer = CountVectorizer(stop_words='english', binary=True)

# First fit the data
vectorizer.fit(data_train.data + data_test.data)

# Then transform it
X_train = vectorizer.transform(data_train.data)
X_test = vectorizer.transform(data_test.data)

# Model
clf = BernoulliNB(alpha=.01) # smoothing parameter, 0 for no
clf.fit(X_train, y_train)

# Predict 
y_predicted = clf.predict(X_test)

score = metrics.accuracy_score(y_test, y_predicted)
print("accuracy: %0.3f" % score)









>2 " Multinomial Naive Bayes Model " > "Better at large vocabulary size"
" A document is considered to be an ordered sequence of word events drawn from the same vocabulary, word frequency counts "

#------------ Start Model Example
from sklearn.datasets import fetch_20newgroups # 20 news groups datasets
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

categories = ['alt.atheism','talk.religion.misc','comp.graphics','sci.space']

# Loading train data
data_train = fetch_20newgroups(subset='train',
	                           categories=categories,
	                           shuffle=True,
	                           random_state=42)

data_test = fetch_20newgroups(subset='test',
	                           categories=categories,
	                           shuffle=True,
	                           random_state=42)
y_train, y_test = data_train.target, data_test.target


# Vectorizing data
feature_extractor_type = "hashed"

if feature_extractor_type == "hashed":
	vectorizer = HashingVectorizer(stop_words='english')
elif feature_extraction_type == "count":
	vectorizer = CountVectorizer(stop_words='english', binary=True)

# First fit the data
vectorizer.fit(data_train.data + data_test.data)

# Then transform it
X_train = vectorizer.transform(data_train.data)
X_test = vectorizer.transform(data_test.data)

# Model
clf = MultinomialNB(alpha=.01) # smoothing parameter, 0 for no
clf.fit(X_train, y_train)

# Predict 
y_predicted = clf.predict(X_test)

score = metrics.accuracy_score(y_test, y_predicted)
print("accuracy: %0.3f" % score)








-1- " Dynamic Bayesian Networks " >>> "States changes with times "

" [TBD] "





-2- " Hidden Markov Model (Special case of DBN) " >>> " Not Markov network but satisfied Markov process - current state only dependes on the previous state "

>1 " Multinomial HMM"
from hmmlearn.hmm import MultinomialHMM
import numpy as np 

model_multinomial = MultinomialHMM(n_components=4) # 4 hidden states

# Transition probability
transition_matrix = np.array([[0.2, 0.6, 0.15, 0.05],
	                          [0.2, 0.3, 0.3, 0.2],
	                          [0.05, 0.05, 0.7, 0.2],
	                          [0.005, 0.045, 0.15, 0.8]])

# Setting the transition prob
model_multinomial.transmat_ = transition_matrix

# Initial state probability
initial_state_prob = np.array([0.1, 0.4, 0.4, 0.1])

# Setting the initial prob
model_multinomial.startprob_ = initial_state_prob

# X - States (emission prob)
emission_prob = np.array([[0.045, 0.15, 0.2, 0.6, 0.005],
	                      [0.2, 0.2, 0.2, 0.3, 0.1],
	                      [0.3, 0.1, 0.1, 0.05, 0.45],
	                      [0.1, 0.1, 0.2, 0.05, 0.55]])

# Start emission prob
model_multinomial.emissionprob_ = emission_prob

Z, X = model_multinomial.sample(100)
# model.sample returns both observations as well as hidden states the first return argument being the observation and the second being the hidden states



>2 " Gaussian HMM "

from hmmlearn.hmm import GaussiamHMM
import matplotlib.pyplot as plt
import numpy as np

model_gaussian = GaussianHMM(n_components=3, covariance_type='full')

transition_matrix = np.array([[0.2, 0.6, 0.2],
	                          [0.4, 0.3, 0.3],
	                          [0.05, 0.05, 0.9]])

model_gaussian.transmat_ = transition_matrix

initial_state_prob = np.array([0.1, 0.4, 0.5])

model_gaussian.startprob_ = initial_state_prob


mean = np.array([[0.0, 0.0],
	             [0.0, 10.0],
	             [10.0, 0.0]])

model_gaussian,means = mean

covariance = 0.5 * np.tile(np.identity(2), (3, 1, 1))

model_gaussian.covars_ = covariance

Z, X = model_gaussian.sample(100)




