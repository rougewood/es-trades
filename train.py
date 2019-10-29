from agent.agent import Agent
from functions import *
import sys
import time
import matplotlib.pyplot as plt

if len(sys.argv) != 4:
    print("Usage: python train.py [stock] [window] [episodes]")
    sys.exit()

stock_name, input_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

agent = Agent(input_size)
# data = getStockDataVec(stock_name)
data = create_trans_bars(stock_name)
l = len(data) - 1
batch_size = 60
epoch_profits = []


for e in range(episode_count):
    tradeLog = open("tradeEpoch"+str(e)+".txt", "a")

    print("Episode " + str(e) + "/" + str(episode_count))
    # state = getState(data, 0, window_size + 1)
    # state = data.iloc[0:5, 1:]
    state = data.iloc[0:input_size, 1:]
    print("state : {}".format(state))
    # time.sleep(4)

    total_profit = 0
    profit_count = 0
    loss_count = 0

    agent.inventory = []

    for t in range(l - input_size):
        tradeLog.write("\n ---epoch : {}--- , ".format(e))

        action = agent.act(state)

        # sit
        # next_state = getState(data, t + 1, input_size + 1)
        next_state = data.iloc[t:t + input_size, 1:]
        # print(type(next_state))
        # next_state = next_state.to_numpy()
        # print(type(next_state))
        # print("next_state: {}".format(next_state))

        # time.sleep(4)
        # print("next_state :{}".format(next_state))

        reward = 0

        if action == 1:  # buy
            agent.inventory.append(data.iloc[t + input_size - 1, 0])
            print("Buy: " + formatPrice(data.iloc[t + input_size - 1, 0]))
            # print("data : {} ".format(data.iloc[t + input_size - 1, :]))
            tradeLog.write("Buy: " + formatPrice(data.iloc[t + input_size - 1, 0]) + ", date : "
                           + str(data.index[t + input_size - 1]))

        elif action == 2 and len(agent.inventory) > 0:  # sell
            bought_price = agent.inventory.pop(0)
            reward = max(data.iloc[t + input_size - 1, 0] - bought_price, 0)
            profit = data.iloc[t + input_size - 1, 0] - bought_price
            total_profit += profit
            tradeLog.write("Sell: " + formatPrice(data.iloc[t + input_size - 1, 0]) + " | Profit: " + formatPrice(
                profit) + ", date : " + str(data.index[t + input_size - 1]))

            if profit>0:
                profit_count += 1
            else:
                loss_count += 1
            print("Sell: " + formatPrice(data.iloc[t + input_size - 1, 0]) + " | Profit: " + formatPrice(profit))

        done = True if t == l - input_size - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")
            epoch_profits.append(total_profit)
            f = open("epoch_profit.txt", "a")
            f.write(" epoch : {} , ".format(e))
            f.write(" profit : {} , ".format(total_profit))
            f.write(" profit/loss : {} \n".format(str(profit_count)+"/"+str(loss_count)))
            f.close()

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    if e % 10 == 0:
        agent.model.save("models/model_ep" + str(e))

    tradeLog.close()


plt.scatter(np.linspace(1, episode_count, num=episode_count), epoch_profits)
plt.xlabel('epoch')
plt.ylabel('profit')
plt.show()
