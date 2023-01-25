# Lab 5 - Open Optical Network ANTOINE POUILLARD
from matplotlib import pyplot as plt
import random
from Network import Network
from Connection import Connection
from tabulate import tabulate


def main():
    network = Network()
    nodeValue = 'ABCDEF'
    signal_power = 0.001

    for i in range(0, 100):
        inputNode = random.choice(nodeValue)
        outputNode = random.choice(nodeValue)
        while inputNode == outputNode:  # if we have the same node
            inputNode = random.choice(nodeValue)
            outputNode = random.choice(nodeValue)
            if inputNode != outputNode:
                break

        connections = Connection(inputNode, outputNode, signal_power)

        # network.stream(connections, 'latency')
        network.stream(connections, 'snr')

    # network.draw()

    df = network.chanel_availability()
    # print(tabulate(df, showindex=True, headers=df.columns))


if "__main__" == __name__:
    main()
