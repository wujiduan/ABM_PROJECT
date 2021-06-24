#Phase Diagram
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from ABM_model import AttendanceModel

#Adj Matrix Visualization
import pathpy as pp
from IPython.display import *
from mesa import Model
from mesa import Agent
import matplotlib.pyplot as plt




##############################################

def AdjacencyGauss(mu, sigma, num_agents): 
    adjacencyMatrix = np.random.normal(mu, sigma, (num_agents, num_agents))
    adjacencyMatrix = (adjacencyMatrix + adjacencyMatrix.T) / 2
    for i in range(num_agents):
        for j in range(num_agents): 
            if adjacencyMatrix[i][j]<0:
                adjacencyMatrix[i][j]=0
            elif adjacencyMatrix[i][j]>1:
                adjacencyMatrix[i][j]=1
    return adjacencyMatrix

##############################################            
    
    
def plot_network(values):
    # get the plot
    # Lets assume: agent we are interest in is 1(red), neighbors are 2(blue), all other agents are 0(white)
        net = values[0]
        colors = values[1]
        html = pp.visualisation.html.generate_html(net,  **{"node_color": colors}) #double ** is a dictionary ex. params['nodecolors'] = colors
        chart = HTML(html)
        display(chart)

def networkneigh(agent,adjMatrix):
    outNeighbors = []
    inNeighbors = []
    
    nagents = adjMatrix.shape[1]
    
    for idx in range(nagents):
        if idx!=agent:
            if adjMatrix[agent,idx] != 0:
                outNeighbors.append(idx)
            if adjMatrix[idx, agent] !=0:
                inNeighbors.append(idx)
    return outNeighbors,inNeighbors

def create_color_net(adjMatrix, pos, noOfAgents):
    net = pp.Network(directed=True)
    colors={}
    outNeighbors,inNeighbors = networkneigh(pos, adjMatrix)
    outnegh, innegh = networkneigh(pos, adjMatrix)
    for node_id in range(noOfAgents):
        if node_id ==pos:
            colors[str(node_id)] = "#FF0000"
        elif node_id in outnegh:
            colors[str(node_id)] = "#0000FF"
        elif node_id in innegh:
            colors[str(node_id)] = "#00FF00"
        net.add_node(str(node_id))
    for source in range(noOfAgents):
        for target in range(noOfAgents):
            if adjMatrix[source,target] == 0:
                continue
            net.add_edge(str(source), str(target))
    return net, colors


def neighbor(obj,pos, noOfAgents):
    
    "Arguments: Adj Matrix, agent of interest, number of agents"

    # initialize a list for neighbors
    neighbors = []
    #adjacency matrix neighbors are determined by the links(networks)
    #and not by proximity, hence we create a random directed network 
    adjMatrix = obj
    net, colors = create_color_net(adjMatrix, pos, noOfAgents)
    agentStates = [net, colors]
    plot_network(agentStates)
    
##############################################

def plot_attendance(model):
    plt.plot(model.datacollector.get_model_vars_dataframe()['Attendance'])
    plt.xlabel('Steps')
    plt.ylabel('Attendance')
    plt.show()
    

    
    
##############################################    
   
def plot_agents(model):
    """ Plot opinion dynamics of all agents 
    Args:
        model 
    """
    
    df=model.datacollector.get_agent_vars_dataframe().reset_index()
    #a dataframe of three columns ["Step", "AgentID", "Opinion"]
    timesteps = df["Step"].unique()
    agent_ids = df["AgentID"].unique()

    # plot each agent's emotion evolution against time steps
    for i in agent_ids:
        emotions_i = df.loc[df.loc[:,"AgentID"] == i, :]["Emotion"].values
        color = emotions_i[0]
        plt.plot(timesteps, emotions_i, "-", alpha=0.3,
                 color=plt.get_cmap('rainbow')(color))
    plt.show()
    
##############################################



def plot_adj_matrix(adjacencyMatrix, num_agents):

    fig, ax = plt.subplots(figsize=(14,14))
    ax.matshow(adjacencyMatrix, cmap=plt.cm.Blues)

    for i in range(num_agents):
        for j in range(num_agents):
            if i==j: 
                adjacencyMatrix[i][j]=0

            c = round(adjacencyMatrix[j,i],2)
            ax.text(i, j, str(c), va='center', ha='center')
            
def get_adj_matrix(model, step): 
    """ Get the adj_matrix at each step
    Arguments:  model, 
                    step : -1 to get the last step matrix"""
    return model.datacollector.get_model_vars_dataframe()['adjacencyMatrix'].iloc[step]

##############################################


def initial_emotion_hist(model, num_agents):
    plt.hist(model.datacollector.get_agent_vars_dataframe().reset_index()["Emotion"].iloc[0:num_agents].to_numpy(), 
            bins=np.arange(0,1,1/10))
    plt.xlabel('Value')
    plt.ylabel('Occurence')
    plt.title("Initial Emotion Distribution")
    plt.show()

###########################################


def plot_attendance_step(model): 
  
    df=model.datacollector.get_agent_vars_dataframe()["Attend"].reset_index()
    timesteps = df["Step"].unique()
    agent_ids = df["AgentID"].unique()

    fig, ax = plt.subplots(figsize=(7, 7))
    # plot each agent's attendance value against time steps
    for i in agent_ids:
        attend_i = df.loc[df.loc[:,"AgentID"] == i, :]["Attend"].values
        color = attend_i
        myarray = np.empty(len(timesteps), dtype=np.int)
        myarray.fill(i)
        cols=[]
        for c in color:
            if c:
                cols.append('green')
            else:
                cols.append('red')
        scatter=ax.scatter(timesteps, myarray, s=1, c=cols, alpha=1)
    plt.show()
    
    
    #if color.any()!= False:
         #   legend1 = ax.legend(*scatter.legend_elements(),
         #                       loc="upper left", title="Attendance")
    #ax.add_artist(legend1)





    
    
    