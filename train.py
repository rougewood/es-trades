from agent.agent import Agent
from functions import *
import sys

if len(sys.argv) != 4:
	print("Usage: python train.py [stock] [window] [episodes]")
	sys.exit()

stock_name, input_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

agent = Agent(input_size)
# data = getStockDataVec(stock_name)
data = create_trans_bars(stock_name)
l = len(data) - 1
batch_size = 32

for e in range(episode_count + 1):
	print ("Episode " + str(e) + "/" + str(episode_count))
	# state = getState(data, 0, window_size + 1)
	state = data.iloc[0, 1:]

	total_profit = 0
	agent.inventory = []

	for t in range(l):
		action = agent.act(state)

		# sit
		# next_state = getState(data, t + 1, input_size + 1)
		next_state = data.iloc[t, 1:]
		reward = 0

		if action == 1: # buy
			agent.inventory.append(data.iloc[t, 0])
			print("Buy: " + formatPrice(data.iloc[t, 0]))

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(data.iloc[t, 0] - bought_price, 0)
			total_profit += data.iloc[t, 0] - bought_price
			print("Sell: " + formatPrice(data.iloc[t, 0]) + " | Profit: " + formatPrice(data.iloc[t, 0] - bought_price))

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print( "--------------------------------")
			print ("Total Profit: " + formatPrice(total_profit))
			print ("--------------------------------")

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	if e % 10 == 0:
		agent.model.save("models/model_ep" + str(e))
