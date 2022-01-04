# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 00:27:52 2021

@author: workstation
"""
import pandas as pd
from queue import Queue
df = pd.read_csv('worldcities.csv')
class Node():
    def __init__(self, city, lat, long):
        self.city = city
        self.lat = lat
        self.long = long


class StackFrontier():
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_city(self, city):
        return any(node.city == city for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node


class QueueFrontier(StackFrontier):

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node