import copy
import numpy as np
import torch

from envs import make_env
from envs.utils import get_goal_distance
from algorithm.replay_buffer import Trajectory, goal_concat
from utils.gcc_utils import gcc_load_lib, c_double, c_int
from deap import algorithms, base, benchmarks, tools, creator
from algorithm.diffusion.visualizer import Visualizer
import time, array, random, copy, math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import deque
from scipy.stats import wasserstein_distance
creator.create("FitnessMin", base.Fitness, weights=(1.0, 1.0, -1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
class TrajectoryPool:
	def __init__(self, args, pool_length):
		self.args = args
		self.length = pool_length

		self.pool = []
		self.pool_init_state = []
		self.counter = 0

	def insert(self, trajectory, init_state):
		if self.counter<self.length:
			self.pool.append(trajectory.copy())
			self.pool_init_state.append(init_state.copy())
		else:
			self.pool[self.counter%self.length] = trajectory.copy()
			self.pool_init_state[self.counter%self.length] = init_state.copy()
		self.counter += 1

	def pad(self):
		if self.counter>=self.length:
			return copy.deepcopy(self.pool), copy.deepcopy(self.pool_init_state)
		pool = copy.deepcopy(self.pool)
		pool_init_state = copy.deepcopy(self.pool_init_state)
		while len(pool)<self.length:
			pool += copy.deepcopy(self.pool)
			pool_init_state += copy.deepcopy(self.pool_init_state)
		return copy.deepcopy(pool[:self.length]), copy.deepcopy(pool_init_state[:self.length])

class MatchSampler:
	def __init__(self, args, achieved_trajectory_pool):
		self.args = args
		self.device = args.device
		self.env = make_env(args)
		self.env_test = make_env(args)
		self.dim = np.prod(self.env.reset()['achieved_goal'].shape)
		self.delta = self.env.distance_threshold
		self.goal_distance = get_goal_distance(args)

		self.length = args.episodes
		init_goal = self.env.reset()['achieved_goal'].copy()
		self.pool = np.tile(init_goal[np.newaxis,:],[self.length,1])+np.random.normal(0,self.delta,size=(self.length,self.dim))
		self.init_state = self.env.reset()['observation'].copy()

		self.match_lib = gcc_load_lib('learner/cost_flow.c')
		self.achieved_trajectory_pool = achieved_trajectory_pool

		if self.args.env == 'FetchPush-v1' or self.args.env == 'FetchPushObs-v1':
			max_action = [1.55, 1.1]
			min_action = [1.05, 0.4]
			visualizer_config = {'number_of_points': 50,
								 'x_min': min_action[0],
								 'y_min': min_action[1],
								 'x_max': max_action[0],
								 'y_max': max_action[1],
								 'env_name': self.args.env,
								 'device': self.device,
								 'sub_folder': self.args.log_subfolder_name
								 }
		elif self.args.env in ['FetchPickAndPlace-v1', 'FetchPickAndPlaceObs-v1', 'FetchReach-v1', 'FetchReachObs-v1']:
			max_action = [1.55, 1.1, 0.4]
			min_action = [1.05, 0.4, 0.9]
			visualizer_config = {'number_of_points': 50,
								 'x_min': min_action[0],
								 'y_min': min_action[1],
								 'x_max': max_action[0],
								 'y_max': max_action[1],
								 'z_min': max_action[2],
								 'z_max': min_action[2],
								 'env_name': self.args.env,
								 'device': self.device,
								 'sub_folder': self.args.log_subfolder_name
								 }

		elif self.args.env == 'FetchSlide-v1' or self.args.env == 'FetchSlideObs-v1':
			max_action = [1.95, 0.8]
			min_action = [0.3, 1.2]
			visualizer_config = {'number_of_points': 50,
								 'x_min': min_action[0],
								 'y_min': min_action[1],
								 'x_max': max_action[0],
								 'y_max': max_action[1],
								 'env_name': self.args.env,
								 'device': self.device,
								 'sub_folder': self.args.log_subfolder_name
								 }

		else:
			raise NotImplementedError

		# self.visualizer = Visualizer(**visualizer_config)
		# estimating diameter
		self.max_dis = 0
		for i in range(1000):
			obs = self.env.reset()
			dis = self.goal_distance(obs['achieved_goal'],obs['desired_goal'])
			if dis>self.max_dis: self.max_dis = dis

	def add_noise(self, pre_goal, noise_std=None):
		goal = pre_goal.copy()
		dim = 2 if self.args.env[:5]=='Fetch' else self.dim
		if noise_std is None: noise_std = self.delta
		goal[:dim] += np.random.normal(0, noise_std, size=dim)
		return goal.copy()

	def sample(self, idx):
		if self.args.env[:5]=='Fetch':
			# return self.add_noise(self.pool[idx])
			return self.pool[idx]
		else:
			return self.pool[idx].copy()#erdi_test

	def find(self, goal):
		res = np.sqrt(np.sum(np.square(self.pool-goal),axis=1))
		idx = np.argmin(res)
		if test_pool:
			self.args.logger.add_record('Distance/sampler', res[idx])
		return self.pool[idx].copy()

	def fitness_function(self, function_input,  desired_goals, aim_discriminator, sol_con):
		with torch.no_grad():
			for ind_index in range(len(function_input)):
				if str(function_input[ind_index]) == "nan":
					function_input[ind_index] = 0

			if self.args.env in ["FetchPickAndPlace-v1", "FetchPickAndPlaceObs-v1", "FetchReach-v1", "FetchReachObs-v1"]:
				solution_goal = function_input
			elif self.args.env == 'FetchPush-v1' or self.args.env == 'FetchPushObs-v1':
				solution_goal = np.hstack((function_input.tolist(), 0.42))
			elif self.args.env == 'FetchSlide-v1' or self.args.env == 'FetchSlideObs-v1':
				solution_goal = np.hstack((function_input.tolist(), 0.41401894))
			else:
				raise NotImplementedError(self)
			#solution_goal = np.tile(solution_goal, (len(achieved_pool_init_state), 1))
			#obs = torch.tensor(np.hstack((np.array(achieved_pool_init_state),solution_goal)),dtype=torch.float32).to("cuda:0")
			obs = torch.tensor(solution_goal, dtype=torch.float32).to(self.args.device).reshape(1,-1)
			obs = self.args.agent.obs_normalizer.normalize(obs)
			actions = self.args.agent.pi(obs)
			value = self.args.agent.q(obs, actions)[:, 0].detach().cpu().numpy()

			distance = np.sqrt(np.sum(np.square(solution_goal[-3:] - np.array(desired_goals)), axis=1))

			fitness_value =  value.max()/ (self.args.hgg_L / self.max_dis / (1 - self.args.gamma))

			discrimiator_output = aim_discriminator.forward(torch.hstack((torch.tensor(np.tile(solution_goal[-3:], (len(desired_goals), 1)), dtype=torch.float32,device=self.device),torch.tensor(np.array(desired_goals), dtype=torch.float32, device=self.device))))

			sol_con.append(solution_goal[-3:])

			distance_dim_x = wasserstein_distance(np.array(sol_con)[:, 0], np.array(desired_goals)[:, 0])
			distance_dim_y = wasserstein_distance(np.array(sol_con)[:, 1], np.array(desired_goals)[:, 1])
			distance_2d = np.sqrt(distance_dim_x ** 2 + distance_dim_y ** 2)


			return discrimiator_output.mean().detach().cpu().tolist(), fitness_value, distance_2d, distance.max()

	def sample_ea_goal(self, initial_goals, desired_goals, aim_discriminator):
		if self.achieved_trajectory_pool.counter==0:
			self.pool = copy.deepcopy(desired_goals)
			return self.pool

		min_x, max_x = np.array(initial_goals)[:,0].min(), np.array(desired_goals)[:,0].max()
		min_y, max_y =  np.array(initial_goals)[:,1].min(), np.array(desired_goals)[:,1].max()
		min_z, max_z = np.array(initial_goals)[:, 2].min(), np.array(desired_goals)[:, 2].max()

		achieved_states, achieved_pool_init_state = self.achieved_trajectory_pool.pad()

		max_states = np.max(np.array(achieved_states), axis=(0, 1))
		min_states = np.min(np.array(achieved_states), axis=(0, 1))

		self.bounds = [(min_states[i], max_states[i]) for i in range(len(max_states))]
		# self.bounds += [(min_x, max_x), (min_y, max_y)]

		if self.args.env in ['FetchReach-v1', 'FetchReachObs-v1']:
			#FetchReach has no object.
			intermediate_goal_limit_1 = list(self.bounds[0])
			intermediate_goal_limit_2 = list(self.bounds[1])
			intermediate_goal_limit_3 = list(self.bounds[2])
		else:
			intermediate_goal_limit_1 = list(self.bounds[3])
			intermediate_goal_limit_2 = list(self.bounds[4])
			intermediate_goal_limit_3 = list(self.bounds[5])
		if intermediate_goal_limit_1[0] < min_x:
			intermediate_goal_limit_1[0] = min_x

		if intermediate_goal_limit_1[1] > max_x:
			intermediate_goal_limit_1[1] = max_x

		if intermediate_goal_limit_2[0] < min_y:
			intermediate_goal_limit_2[0] = min_y

		if intermediate_goal_limit_2[1] > max_y:
			intermediate_goal_limit_2[1] = max_y

		if intermediate_goal_limit_3[0] < min_z:
			intermediate_goal_limit_3[0] = min_z

		if intermediate_goal_limit_3[1] > max_z:
			intermediate_goal_limit_3[1] = max_z

		self.bounds.append(intermediate_goal_limit_1)
		self.bounds.append(intermediate_goal_limit_2)

		if self.args.env in ['FetchPickAndPlace-v1', 'FetchPickAndPlaceObs-v1','FetchReach-v1','FetchReachObs-v1']:
			self.bounds.append(intermediate_goal_limit_3)

		toolbox = base.Toolbox()
		min_limit = []
		max_limit = []
		for b in range(len(self.bounds)):
			min_limit.append(self.bounds[b][0])
			max_limit.append(self.bounds[b][1])

		def uniform(low, up, size=None):
			return [random.uniform(a, b) for a, b in zip(low, up)]

		my_solution_container = deque(maxlen=self.args.deque_size)

		desired_goal_mean = np.array(desired_goals).mean(axis=0)
		desired_goals_clone = desired_goals.copy()
		# for _ in range(50):
		# 	desired_goals_clone.append(desired_goal_mean + [0, np.random.uniform(-0.005, 0.005), 0])

		toolbox.register("attr_float", uniform, min_limit, max_limit)
		toolbox.register("evaluate", self.fitness_function,desired_goals=desired_goals_clone, aim_discriminator=aim_discriminator, sol_con = my_solution_container)
		toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=min_limit, up=max_limit, eta=20.0)
		toolbox.register("mutate", tools.mutPolynomialBounded, low=min_limit, up=max_limit, eta=20.0, indpb=1.0 / 13)
		toolbox.register("select", tools.selNSGA2)
		population_size = self.args.population_size
		generations = self.args.generation_size

		stats = tools.Statistics(lambda ind: ind.fitness.values)
		stats.register("min", np.min, axis=0)
		stats.register("max", np.max, axis=0)
		logbook = tools.Logbook()
		logbook.header = "gen", "evals", "std", "min", "avg", "max"
  

		obstacle_size = self.args.obstacle_size
		def valid(individual):
			"""Determines if the individual is valid or not."""
			if self.args.env in ["FetchPushObs-v1", "FetchSlideObs-v1"] :
				obstacle_pos = np.array(self.args.obstacle_pos[:2])
				max_boundary_box = obstacle_pos + obstacle_size[:2]
				min_boundary_box = obstacle_pos - obstacle_size[:2]
				return_flag = ~(((individual[-2] < max_boundary_box[0]) & (individual[-2] > min_boundary_box[0])) & ((individual[-1] < max_boundary_box[1]) & (individual[-1] > min_boundary_box[1])))
				return return_flag

			elif self.args.env in ["FetchPickAndPlaceObs-v1", "FetchReachObs-v1"]:
				obstacle_pos = np.array(self.args.obstacle_pos)
				max_boundary_box = obstacle_pos + obstacle_size[:3]
				min_boundary_box = obstacle_pos - obstacle_size[:3]
				return_flag = ~(((individual[-3] < max_boundary_box[0]) & (individual[-3] > min_boundary_box[0])) & ((individual[-2] < max_boundary_box[1]) & (individual[-2] > min_boundary_box[1])) & ((individual[-1] < max_boundary_box[2]) & (individual[-1] > min_boundary_box[2])))
				return return_flag

			elif self.args.env in ["FetchPickAndPlace-v1", "FetchPush-v1", "FetchSlide-v1", "FetchReach-v1"]:
				return True
			else:
				raise NotImplementedError
		pop = toolbox.population(n=population_size)

		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in pop if not ind.fitness.valid]
		fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		pop = toolbox.select(pop, len(pop))
		CXPB = 0.7
		for gen in range(1, generations):
			# Vary the population
			offspring = tools.selTournamentDCD(pop, len(pop))
			offspring = [toolbox.clone(ind) for ind in offspring]

			for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
				if random.random() <= CXPB:
					toolbox.mate(ind1, ind2)

				toolbox.mutate(ind1)
				toolbox.mutate(ind2)
				del ind1.fitness.values, ind2.fitness.values

			# Evaluate the individuals with an invalid fitness
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
			fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
			for ind, fit in zip(invalid_ind, fitnesses):
				ind.fitness.values = fit
			# Select the next generation population

			selected_individuals = [ind for ind in pop + offspring if valid(ind)]

			while len(selected_individuals) < population_size:
				tmp_selected_individuals = toolbox.clone(selected_individuals)
				for ind1, ind2 in zip(tmp_selected_individuals[::2], tmp_selected_individuals[1::2]):
					if random.random() <= CXPB:
						toolbox.mate(ind1, ind2)
					toolbox.mutate(ind1)
					toolbox.mutate(ind2)
					del ind1.fitness.values, ind2.fitness.values
				selected_individuals += [ind for ind in tmp_selected_individuals if valid(ind)]
				invalid_ind = [ind for ind in selected_individuals if not ind.fitness.valid]
				fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
				for ind, fit in zip(invalid_ind, fitnesses):
					ind.fitness.values = fit

			# Select the next generation population
			pop = toolbox.select(selected_individuals, population_size)
			record = stats.compile(pop)
			logbook.record(gen=gen, evals=len(invalid_ind), **record)
			print(logbook.stream)



		pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

		if len(pareto_front) < self.args.episodes:
			pareto_front_array = [pareto_front[i][-3:] for i in range(len(pareto_front))]
			for i in range(self.args.episodes - len(pareto_front_array)):
				pareto_front_array.append(pop[i][-3:])
			return np.array(pareto_front_array)

		kmeans = KMeans(n_clusters=self.args.episodes)
		kmeans.fit(pareto_front)
		labels = kmeans.labels_
		centers = kmeans.cluster_centers_



		np.save("log/"+self.args.log_subfolder_name + "/centers" + str(self.args.episode_counter) + ".npy", centers)

		plt.figure(figsize=(5, 5))
		for ind in pop:
			plt.scatter(ind.fitness.values[-2], ind.fitness.values[-1], color='black', edgecolors='black', s=50)
		for i, ind in enumerate(pareto_front):
			color = plt.get_cmap('rainbow')(labels[i] / kmeans.n_clusters)
			plt.scatter(ind.fitness.values[-2], ind.fitness.values[-1], color=color, edgecolors='k', s=50)

		best_value = -100
		for i in range(len(centers)):
			fitness_of_centers = toolbox.evaluate(centers[i])
			plt.scatter(fitness_of_centers[0], fitness_of_centers[1], c='black', marker='X', edgecolors='red', s=200,
						label='Centroids')
			if fitness_of_centers[1] > best_value:
				best_value = fitness_of_centers[1]
				intermediate_goal_index = i
		plt.title('Pareto-optimal front')
		plt.xlabel('distance')
		plt.ylabel('$V$')
		plt.savefig("log/"+self.args.log_subfolder_name + "/pareto_fron" + str(self.args.episode_counter) + ".png")

		# intermediate_goal = np.hstack((pareto_front[0][-2:].tolist(), 0.42))
		# intermediate_goal_index = np.random.randint(centers.shape[0])
		# intermediate_goal =centers[:,-3:]

		if self.args.env in ["FetchPickAndPlaceObs-v1", "FetchReachObs-v1"]:
			obstacle_pos = np.array(self.args.obstacle_pos)
			max_boundary_box = obstacle_pos + obstacle_size[:3]
			min_boundary_box = obstacle_pos - obstacle_size[:3]
			# Watch out there is a bitwise NOT ~ operatator in front of the boolean numpy array below line;
			non_valid_kmeans = np.array(((centers[:,-3] < max_boundary_box[0]) & (centers[:,-3] > min_boundary_box[0])) & ((centers[:,-2] < max_boundary_box[1]) & (centers[:,-2] > min_boundary_box[1])) & (centers[:,-1] < max_boundary_box[2]) & (centers[:,-1] > min_boundary_box[2]))
			while non_valid_kmeans.any():
				centers[non_valid_kmeans, -3:] += np.random.normal(0, 0.05, size=3)
				non_valid_kmeans = np.array(((centers[:, -3] < max_boundary_box[0]) & (centers[:, -3] > min_boundary_box[0])) & ((centers[:, -2] < max_boundary_box[1]) & (centers[:, -2] > min_boundary_box[1])) & (centers[:, -1] < max_boundary_box[2]) & (centers[:, -1] > min_boundary_box[2]))

			intermediate_goal = centers[:, -3:]
		elif self.args.env in ['FetchPushObs-v1','FetchSlideObs-v1']:
			obstacle_pos = np.array(self.args.obstacle_pos[:2])
			max_boundary_box = obstacle_pos + obstacle_size[:2]
			min_boundary_box = obstacle_pos - obstacle_size[:2]
			non_valid_kmeans = np.array(((centers[:,-2] < max_boundary_box[0]) & (centers[:,-2] > min_boundary_box[0])) & ((centers[:,-1] < max_boundary_box[1]) & (centers[:,-1] > min_boundary_box[1])))
			while non_valid_kmeans.any():
				centers[non_valid_kmeans, -2:] += np.random.normal(0, 0.05, size=2)
				non_valid_kmeans = np.array(((centers[:, -2] < max_boundary_box[0]) & (centers[:, -2] > min_boundary_box[0])) & ((centers[:, -1] < max_boundary_box[1]) & (centers[:, -1] > min_boundary_box[1])))
			intermediate_goal = np.hstack((centers[:, -2:], np.tile(desired_goals[0][2], (centers.shape[0], 1))))

		elif self.args.env == "FetchPickAndPlace-v1" or self.args.env == "FetchReach-v1":
			intermediate_goal = centers[:, -3:]
		elif self.args.env == "FetchPush-v1" or self.args.env == "FetchSlide-v1":
			intermediate_goal = np.hstack((centers[:, -2:], np.tile(desired_goals[0][2], (centers.shape[0], 1))))
		else:
			raise NotImplementedError

		#self.visualizer.aim_reward_visualizer(aim_discriminator, self.args.episode_counter, desired_goals)

		return intermediate_goal


	def update(self, initial_goals, desired_goals):
		if self.achieved_trajectory_pool.counter==0:
			self.pool = copy.deepcopy(desired_goals)
			return

		achieved_states, achieved_pool_init_state = self.achieved_trajectory_pool.pad()
		achieved_pool = np.array(achieved_states)[:,:,3:6]
		candidate_goals = []
		candidate_edges = []
		candidate_id = []
		agent = self.args.agent
		achieved_value = []
		for i in range(len(achieved_pool)):
			obs = [ goal_concat(achieved_pool_init_state[i], achieved_pool[i][j]) for  j in range(achieved_pool[i].shape[0])]
			obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
			obs = agent.obs_normalizer.normalize(obs)
			actions = agent.pi(obs)
			value = agent.q(obs,actions)[:,0].detach().cpu().numpy()
			value = np.clip(value, -1.0/(1.0-self.args.gamma), 0)
			achieved_value.append(value.copy())
		n = 0
		graph_id = {'achieved':[],'desired':[]}
		for i in range(len(achieved_pool)):
			n += 1
			graph_id['achieved'].append(n)
		for i in range(len(desired_goals)):
			n += 1
			graph_id['desired'].append(n)
		n += 1
		self.match_lib.clear(n)

		for i in range(len(achieved_pool)):
			self.match_lib.add(0, graph_id['achieved'][i], 1, 0)
		for i in range(len(achieved_pool)):
			for j in range(len(desired_goals)):

				res = np.sqrt(np.sum(np.square(achieved_pool[i]-desired_goals[j]),axis=1)) - achieved_value[i]/(self.args.hgg_L/self.max_dis/(1-self.args.gamma))
				match_dis = np.min(res)+self.goal_distance(achieved_pool[i][0], initial_goals[j])*self.args.hgg_c
				match_idx = np.argmin(res)

				edge = self.match_lib.add(graph_id['achieved'][i], graph_id['desired'][j], 1, c_double(match_dis))
				candidate_goals.append(achieved_pool[i][match_idx])
				candidate_edges.append(edge)
				candidate_id.append(j)
		for i in range(len(desired_goals)):
			self.match_lib.add(graph_id['desired'][i], n, 1, 0)

		match_count = self.match_lib.cost_flow(0,n)
		assert match_count==self.length

		explore_goals = [0]*self.length
		for i in range(len(candidate_goals)):
			if self.match_lib.check_match(candidate_edges[i])==1:
				explore_goals[candidate_id[i]] = candidate_goals[i].copy()
		assert len(explore_goals)==self.length
		self.pool = np.array(explore_goals)

class EALearner:
	def __init__(self, args):
		self.args = args
		self.env = make_env(args)
		self.env_test = make_env(args)
		self.goal_distance = get_goal_distance(args)
		self.device = self.args.device

		self.env_List = []
		for i in range(args.episodes):
			self.env_List.append(make_env(args))

		self.achieved_trajectory_pool = TrajectoryPool(args, args.hgg_pool_size)
		self.sampler = MatchSampler(args, self.achieved_trajectory_pool)

	def learn(self, args, env, env_test, agent, buffer, aim_discriminator):
		initial_goals = []
		desired_goals = []
		env_goal_temporary_container = []
		obs_temporary_container = []
		for i in range(args.episodes):
			obs = self.env_List[i].reset()
			goal_a = obs['achieved_goal'].copy()
			goal_d = obs['desired_goal'].copy()
			initial_goals.append(goal_a.copy())
			desired_goals.append(goal_d.copy())

		self.sampler.update(initial_goals, desired_goals)
		explore_goal_container = self.sampler.sample_ea_goal(initial_goals, desired_goals, aim_discriminator)
		achieved_trajectories = []
		achieved_init_states = []
		achieved_states = []
		for i in range(args.episodes):
			obs = self.env_List[i].get_obs()
			init_state = obs['observation'].copy()
			# explore_goal = self.sampler.sample(i)
			explore_goal = explore_goal_container[i]
			self.env_List[i].goal = explore_goal.copy()
			obs = self.env_List[i].get_obs()
			current = Trajectory(obs)
			trajectory = [obs['achieved_goal'].copy()]
			traj_achieved_states = [obs['observation'].copy()]
			for timestep in range(args.timesteps):
				action = agent.step(obs, explore=True)
				obs, reward, done, info = self.env_List[i].step(action)
				trajectory.append(obs['achieved_goal'].copy())
				traj_achieved_states.append(obs['observation'].copy())
				if timestep==args.timesteps-1: done = True
				current.store_step(action, obs, reward, done)
				if done: break
			env_goal_temporary = explore_goal
			env_goal_temporary_container.append(env_goal_temporary)
			achieved_trajectories.append(np.array(trajectory))
			achieved_states.append(np.array(traj_achieved_states))
			achieved_init_states.append(init_state)
			buffer.store_trajectory(current)
			agent.normalizer_update(buffer.sample_batch())
			obs_temporary_container.append(traj_achieved_states)
			self.args.episode_counter += 1

			if buffer.steps_counter>=args.warmup:
				for _ in range(args.train_batches):
					info = agent.train(buffer.sample_batch())
					args.logger.add_dict(info)
				agent.target_update()

		selection_trajectory_idx = {}
		for i in range(self.args.episodes):
			if self.goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1])>0.01:
				selection_trajectory_idx[i] = True
		for idx in selection_trajectory_idx.keys():
			self.achieved_trajectory_pool.insert(achieved_states[idx].copy(), achieved_init_states[idx].copy())

		for _ in range(1):
			sample_buffer = buffer.sample_batch()
			if self.args.env =="FetchReach-v1":
				achieved_states_tensor = torch.tensor(np.array(np.array(sample_buffer['obs'])[:, 0:3]),dtype=torch.float32).to(self.device)
				achieved_states_tensor_next = torch.tensor(np.array(np.array(sample_buffer['obs_next'])[:, 0:3]),dtype=torch.float32).to(self.device)
			else:
				achieved_states_tensor = torch.tensor(np.array(np.array(sample_buffer['obs'])[:,3:6]),dtype=torch.float32).to(self.device)
				achieved_states_tensor_next = torch.tensor(np.array(np.array(sample_buffer['obs_next'])[:, 3:6]),dtype=torch.float32).to(self.device)

			# desired_states_tensor  = torch.tile(torch.tensor(np.array(desired_goals[0]), dtype=torch.float32).to(self.device),[achieved_states_tensor.shape[0], 1])
			desired_states_tensor = torch.tensor(np.tile(desired_goals, (achieved_states_tensor.shape[0] // len(desired_goals) + 1, 1))[:achieved_states_tensor.shape[0]], dtype=torch.float32, device=self.device)
			policy_states = torch.cat([achieved_states_tensor,desired_states_tensor], dim=-1)
			policy_next_states = torch.cat([achieved_states_tensor_next,desired_states_tensor], dim=-1)
			target_states = torch.cat([desired_states_tensor + torch.from_numpy(np.random.normal(scale=0.01, size=desired_states_tensor.shape)).float().to(self.device),desired_states_tensor], dim=-1)  # s_g, s_g
			aim_disc_loss, wgan_loss, graph_penalty, min_aim_f_loss = aim_discriminator.optimize_discriminator(target_states, policy_states, policy_next_states)


		np.save("log/" + self.args.log_subfolder_name + "/trajectory_train/env_goals" + str(int(self.args.episode_counter/args.episodes)-1) + ".npy", np.array(env_goal_temporary_container))
		np.save("log/" + self.args.log_subfolder_name + "/trajectory_train/train_obs" + str(int(self.args.episode_counter / args.episodes) - 1) + ".npy", np.array(obs_temporary_container))

		torch.save(aim_discriminator.state_dict(),"log/" + self.args.log_subfolder_name + "/debug/discriminator" + str(int(self.args.episode_counter / args.episodes) - 1) + ".pth")
