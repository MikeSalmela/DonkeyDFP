from doomrunnermulti import  Runner
from DFPmulti import DFPAgent
import threading

def runEpisodes(runner):
    for episode in range(25):
        runner.runEpisode()

imgShape = (84, 84)
runner_count = 2
agent = DFPAgent(6, (*imgShape, 2), (3,), (3*6,), memc=runner_count)

runners = []


for i in range(runner_count):
    runners.append(Runner(agent, i==0, ind=i))

runners[0].runEpisode()
runners[0].runEpisode()

for i in range(50000):
    for runner in runners:
        x = threading.Thread(target=runEpisodes, args=(runner,))
        x.start()
    x.join()
    agent.decayLearningRate()
    for runner in runners:
        all_steps = runner.m_steps
        print(f"Episodes: {runner.episodes}")
    runners[0].save()

    print(f"steps: {all_steps}")
