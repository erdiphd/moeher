from common import get_args,experiment_setup
from algorithm.replay_buffer import goal_based_process
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from envs import make_env

if __name__=='__main__':
	args = get_args()
	env, env_test, agent, buffer, learner, tester, aim_discriminator = experiment_setup(args)

	path = "/media/erdi/erdihome_hdd/Codes/HER_EA/training_results/without_gpu/her_ea/log/20240105000130_e7490"
	aim_discriminator.load_state_dict(torch.load(path + '/debug/discriminator399.pth'))

	env_List = []
	for i in range(args.episodes):
		env_List.append(make_env(args))

	initial_goals = []
	desired_goals = []
	env_goal_temporary_container = []
	obs_temporary_container = []
	for i in range(args.episodes):
		obs = env_List[i].reset()
		goal_a = obs['achieved_goal'].copy()
		goal_d = obs['desired_goal'].copy()
		initial_goals.append(goal_a.copy())
		desired_goals.append(goal_d.copy())


	max_action = [1.55, 1.1]
	min_action = [1.05, 0.4]
	visualizer_config = {'number_of_points': 50,
						 'x_min': min_action[0],
						 'y_min': min_action[1],
						 'x_max': max_action[0],
						 'y_max': max_action[1],
						 'env_name': args.env,
						 'device': args.device,
						 'sub_folder': args.log_subfolder_name
						 }
	x_min, x_max = visualizer_config['x_min'], visualizer_config['x_max']
	y_min, y_max = visualizer_config['y_min'], visualizer_config['y_max']
	number_of_points = visualizer_config['number_of_points']
	x_debug = torch.linspace(x_min, x_max, number_of_points)
	y_debug = torch.linspace(y_min, y_max, number_of_points)
	X_debug, Y_debug = torch.meshgrid(x_debug, y_debug)
	x_input_debug = X_debug.reshape(-1, 1)
	y_input_debug = Y_debug.reshape(-1, 1)
	if args.env in ["FetchPush-v1", "FetchSlide-v1"]:
		z_input_debug = torch.tile(torch.tensor([0.42469975]), [len(x_input_debug), 1])
	else:
		raise NotImplementedError
	with_respect_to = np.array(desired_goals)
	virtual_diffusion_goals_debug = torch.hstack((x_input_debug, y_input_debug, z_input_debug)).to(args.device)
	with_respect_to = torch.tensor(np.tile(with_respect_to, (virtual_diffusion_goals_debug.shape[0] // len(with_respect_to) + 1, 1))[:virtual_diffusion_goals_debug.shape[0]], dtype=torch.float32, device=args.device)
	inputs_norm_tensor_tmp = torch.hstack((virtual_diffusion_goals_debug, with_respect_to))
	aim_reward = aim_discriminator.forward(inputs_norm_tensor_tmp)



	fig, ax = plt.subplots()
	x = x_debug.numpy()
	y = y_debug.numpy()
	V = aim_reward.detach().cpu().numpy().reshape(x.shape[0],y.shape[0])
	c = ax.pcolormesh(x, y, V, cmap='RdBu_r')
	# map = Map()
	# ax.plot(map.ox, map.oy, ".k")
	# fig.colorbar(c, orientation='vertical')
	# ax.set(xlim=[-6, 14], ylim=[-6, 14])
	# ax.set_aspect('equal', 'box')
	ax.scatter(x=with_respect_to[:, 0], y=with_respect_to[:, 1], s=100, color='white',edgecolors='black')
	ax.scatter(x=np.array(initial_goals)[:, 0], y=np.array(initial_goals)[:, 1], s=100, color='yellow', edgecolors='black')

	index_number = 0
	cmap = plt.get_cmap('cool')
	intermediate_goals_total = []
	alphas = np.linspace(0, 1, 21)
	for ind_i, i in enumerate(range(index_number, 400,20)):
		intermediate_goals = np.load(path + "/trajectory_train/env_goals" + str(i) + ".npy", allow_pickle=True)
		intermediate_goals = intermediate_goals.reshape(-1, 3)
		rgb = tuple(np.array(cmap(float(alphas[ind_i]))[:3]))
		ax.scatter(x=intermediate_goals[:, 0], y=intermediate_goals[:, 1], color=rgb, edgecolor='k', label='Barycenter',alpha=0.6)
		intermediate_goals_total.append(intermediate_goals)

	print('test')
