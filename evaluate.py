import keras
from keras.models import load_model

from agent.agent import Agent
from functions import *
import sys
import time

if len(sys.argv) != 3:
	print ("Usage: python evaluate.py [stock] [model]")
	exit()

stock_name, model_name = sys.argv[1], sys.argv[2]
model = load_model("models/" + model_name)
input_size = model.layers[0].input.shape.as_list()[1]
print("input_size {}".format(input_size))

agent = Agent(input_size, True, model_name)
# data = getStockDataVec(stock_name)
data = create_trans_bars(stock_name)
l = len(data) - 1
print("length l = {}".format(l))
batch_size = 32

# state = getState(data, 0, input_size + 1)
state = data.iloc[0:input_size, 1:]
print("state  =  {}".format(state))

total_profit = 0
agent.inventory = []

for t in range(l):
	action = agent.act(state)
	print("action : {}".format(action))
	# time.sleep(1)

	# sit
	# next_state = getState(data, t + 1, input_size + 1)
	next_state = data.iloc[t:t+input_size, 1:]
	print("next_state :{}".format(next_state))
	reward = 0

	if action == 1: # buy
		agent.inventory.append(data.iloc[t:t+input_size, 1:])
		print ("Buy: " + formatPrice(data.iloc[t:t+input_size, 0]))

	elif action == 2 and len(agent.inventory) > 0: # sell
		bought_price = agent.inventory.pop(0)
		profit = data.iloc[t+input_size-1, 0] - bought_price
		reward = max(profit, 0)
		total_profit += profit
		print("Sell: " + formatPrice(data.iloc[t+input_size-1, 0]) + " | Profit: " + profit)

	done = True if t == l - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state

	if done:
		print ("--------------------------------")
		print (stock_name + " Total Profit: " + formatPrice(total_profit))
		print ("--------------------------------")
