import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
import math

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, post_step_callback=None,
                 shared_viewer=True, discrete_action=False):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.post_step_callback = post_step_callback
        # environment parameters
        self.discrete_action_space = discrete_action
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,))
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            c_action_space = spaces.Discrete(world.dim_c)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = spaces.MultiDiscrete([[0,act_space.n-1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,)))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def _seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def _step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        #print("action_n:",action_n)
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n
        if self.post_step_callback is not None:
            self.post_step_callback(self.world)
        return obs_n, reward_n, done_n, info_n

    def _reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        #print("action:",action)
        #action: [0. 0. 0. 1. 0.]
        agent.action.u = np.zeros(self.world.dim_p)
        #agent.action.u = array([0.,0.])
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        action = [action]
        #print("now_action:",action)
        #now_action: [array([0., 0., 0., 1., 0.], dtype=float32)]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
            else:
                #print("action[0]:",action[0][0])
                #action[0]: [[ 0.  1.],[-1.  0.],[ 0. -1.],[ 1.  0.]]
                if self.discrete_action_space:
                    #True
                    #print("turn_n:",self.turn_n)
                    #print("agent.i:",agent.i)
                    #print("agent%d t_i:"%agent.i,self.t_i[agent.i])
                    if(agent.holding == None and agent.recover):
                        if(agent.turn_n == 0):
                            if(agent.t_i == (agent.turn_n+1)*6+agent.i):
                                agent.turn_n += 1
                        elif(agent.turn_n <= 2):
                            if(agent.t_i == (agent.turn_n+1)*6+agent.i+2*agent.turn_n*agent.i):
                                agent.turn_n += 1
                        elif (agent.turn_n % 2 == 1):
                            if(agent.t_i == ((agent.turn_n+1)*6+agent.i+(((agent.turn_n-1)**2/4)*3)+2*agent.turn_n*agent.i)):
                                agent.turn_n += 1
                        else:
                            if(agent.t_i == ((agent.turn_n+1)*6+agent.i+(agent.turn_n/2-1)*(agent.turn_n/2)*3+2*agent.turn_n*agent.i)):
                                agent.turn_n += 1
                        agent.t_i += 1
                        agent.action.u += action[0][(agent.turn_n%4)]
                    elif (agent.holding == None and not agent.recover):
                        #print("linalg:",np.linalg.norm(agent.state.rep_pos))
                        agent.action.u += np.array([(agent.state.rep_pos[0]/np.linalg.norm(agent.state.rep_pos))
                                                   ,(agent.state.rep_pos[1]/np.linalg.norm(agent.state.rep_pos))])
                        if((math.fabs(agent.state.p_pos[0] - agent.state.rep_pos[0]) < 0.0000001) and 
                            (math.fabs(agent.state.p_pos[1] - agent.state.rep_pos[1]) < 0.0000001)):
                            agent.state.rep_pos = np.zeros(self.world.dim_p)
                            agent.recover = True
                        #print("agent%d.state.rep_pos:"%agent.i,agent.state.rep_pos)
                        #print("agent%d.state.p_pos:"%agent.i,agent.state.p_pos)
                    else:
                        #self.turn_n[agent.i] = 0
                        if(agent.recover):
                            agent.state.rep_pos = np.zeros(self.world.dim_p)
                            agent.state.rep_pos += agent.state.p_pos
                            agent.recover = False
                        #print("agent%d.state.rep_pos:"%agent.i,agent.state.rep_pos)
                        #print("agent%d.state.p_pos:"%agent.i,agent.state.p_pos)
                        agent.action.u -= np.array([(agent.state.rep_pos[0]/np.linalg.norm(agent.state.rep_pos))
                                                   ,(agent.state.rep_pos[1]/np.linalg.norm(agent.state.rep_pos))])

                    
                    #print("agent.action.u:",agent.action.u)
                    #agent.action.u += action[0][0]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                #True
                sensitivity = agent.accel
            #print("sensitivity:",sensitivity,"agent.action.u:",agent.action.u)
            agent.action.u *= sensitivity
            #print("new_agent.action.u:",agent.action.u)
            action = action[1:]
            #print("new_action:",action)
            #new_action: []

        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def _render(self, mode='human', close=True):
        if close:
            # close any existic renderers
            for i,viewer in enumerate(self.viewers):
                if viewer is not None:
                    viewer.close()
                self.viewers[i] = None
            return []

        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                #显示屏的大小
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            self.comm_geoms = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                entity_comm_geoms = []
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                    if not entity.silent:
                        dim_c = self.world.dim_c
                        # make circles to represent communication
                        for ci in range(dim_c):
                            comm = rendering.make_circle(entity.size / dim_c)
                            comm.set_color(1, 1, 1)
                            comm.add_attr(xform)
                            offset = rendering.Transform()
                            comm_size = (entity.size / dim_c)
                            offset.set_translation(ci * comm_size * 2 -
                                                   entity.size + comm_size, 0)
                            comm.add_attr(offset)
                            entity_comm_geoms.append(comm)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
                self.comm_geoms.append(entity_comm_geoms)
            for wall in self.world.walls:
                corners = ((wall.axis_pos - 0.5 * wall.width, wall.endpoints[0]),
                           (wall.axis_pos - 0.5 * wall.width, wall.endpoints[1]),
                           (wall.axis_pos + 0.5 * wall.width, wall.endpoints[1]),
                           (wall.axis_pos + 0.5 * wall.width, wall.endpoints[0]))
                if wall.orient == 'H':
                    corners = tuple(c[::-1] for c in corners)
                geom = rendering.make_polygon(corners)
                if wall.hard:
                    geom.set_color(*wall.color)
                else:
                    geom.set_color(*wall.color, alpha=0.5)
                self.render_geoms.append(geom)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)
                for entity_comm_geoms in self.comm_geoms:
                    for geom in entity_comm_geoms:
                        viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
                if 'agent' in entity.name:
                    self.render_geoms[e].set_color(*entity.color, alpha=0.5)
                    if not entity.silent:
                        for ci in range(self.world.dim_c):
                            color = 1 - entity.state.c[ci]
                            self.comm_geoms[e][ci].set_color(color, color, color)
                else:
                    self.render_geoms[e].set_color(*entity.color)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def _step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def _reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def _render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
