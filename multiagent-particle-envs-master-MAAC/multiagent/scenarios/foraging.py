import numpy as np
import copy
from multiagent.core import World, Agent, Landmark,Nest
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 10
        num_agents = 3
        num_landmarks = 10
        num_nest = 1
        world.collaborative = True  # whether agents share rewards
        # add agents
        world.agents = [Agent() for i in  range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.size = 0.04
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        print("world.landmarks:",world.landmarks)
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
        #add nest
        world.nests =  [Nest() for i in range(num_nest)]
        for i, nest in enumerate(world.nests):
            nest.name = 'nest %d' % i
            nest.collide = False
            nest.movable = False
            nest.size = 0.15
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        '''
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want other agent to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        world.agents[1].goal_a = world.agents[0]
        world.agents[1].goal_b = np.random.choice(world.landmarks)
        '''
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])       
            agent.foraging_capability = True        
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.becaught = False
         # random properties for nests
        for i, nest in enumerate(world.nests):
            nest.color = np.array([0.78,0.04,0.25])  
        
       
        '''
        # special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color                
        world.agents[1].goal_a.color = world.agents[1].goal_b.color       
        '''                        
        # set random initial states
        for i, nest in enumerate(world.nests):
            #nest.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            nest.state.p_pos = [0,0]
            nest.state.p_vel = np.zeros(world.dim_p)
        for agent in world.agents:
            #agent.state.p_pos = copy.deepcopy(nest.state.p_pos)
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            #print("agent position",agent.state.p_pos)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        

    def benchmark_data(self, agent, world):
        input("here")
        rew = 0
        collisions = 0
        gotten_targets = 0
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
            
            for l in world.landmarks:
                if self.is_collision(l,agent):
                    if (agent.foraging_capability == True and l.becaught == False):
                        agent.foraging_capability = False
                        l.becaught = True
                        l.color = np.array([1,1,1]) 
                        rew += 1
                        break
            
            for n in world.nests:
                if self.is_collision(n,agent):
                    if agent.foraging_capability == False:
                        agent.foraging_capability = True
                        rew += 100
                        gotten_targets += 1
        return (rew, collisions, gotten_targets)

    def is_collision(self, agent1, agent2):
        #print("agent1 position",agent1.state.p_pos)
        #print("agent2 position",agent2.state.p_pos)
        element_delta_pos = agent1.state.p_pos - agent2.state.p_pos
        #print("delta_pos:", element_delta_pos)
        dist = np.sqrt(np.sum(np.square( element_delta_pos)))
        #print("dist:",dist)
        dist_min = agent1.size + agent2.size
        #print("dist_min:",dist_min)
        return True if dist < dist_min else False


    def reward(self, agent, world):
        '''
        if agent.goal_a is None or agent.goal_b is None:
            return 0.0
        dist2 = np.sum(np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos))
        return -dist2
        '''
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        '''
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        '''
        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        if agent.foraging_capability:
            for l in world.landmarks:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
                rew -= min(dists)
        if not (agent.foraging_capability):
            for n in world.nests:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - n.state.p_pos))) for a in world.agents]
                rew -= min(dists)

        if agent.collide:
            '''
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
            '''
            for l in world.landmarks:
                #print( "collision2",self.is_collision(l,agent))
                if self.is_collision(l,agent):
                    #print("collision l&a")
                    #print("agent",agent.name,"before agent.foraging_capability:",agent.foraging_capability)
                   # input()
                    if (agent.foraging_capability == True and l.becaught == False):
                        agent.foraging_capability = False
                       # print("agent",agent.name,"after agent.foraging_capability:",agent.foraging_capability)
                        #input()
                        #l.state.p_pos = agent.state.p_pos
                        l.becaught = True
                        l.color = np.array([1,1,1]) 
                        rew += 1
                        #input()
                        #print(l.name)
                        #world.landmarks.remove(l)
                        #print("now world.landmarks:",world.landmarks)
                        break
            
            for n in world.nests:
                #print( "collision3",self.is_collision(n,agent))
                if self.is_collision(n,agent):
                    #print("collision n&a")
                    #input()
                    if agent.foraging_capability == False:
                        agent.foraging_capability = True
                        rew += 100
        
        return rew

    def observation(self, agent, world):
        # goal color
        goal_color = [np.zeros(world.dim_color), np.zeros(world.dim_color)]
        '''
        if agent.goal_b is not None:
            goal_color[1] = agent.goal_b.color 
        '''

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
        return np.concatenate([agent.state.p_vel] + entity_pos + [goal_color[1]] + comm)
