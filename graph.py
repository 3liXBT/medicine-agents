from dataclasses import dataclass

import yaml
import streamlit as st


@dataclass(frozen=True)
class Prompt:
    text: str
    parameters: list

    def apply(self, args: dict) -> str:
        text = self.text
        for p in self.parameters:
            text = text.replace(f"{{{{{p}}}}}", args[p])
        return text

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(data["text"], data["parameters"])

    def __hash__(self):
        return hash(self.text)


class ExecutionContext:
    def __init__(self):
        self.context = {}

    def set(self, key, value):
        self.context[key] = value

    def get(self, key):
        return self.context[key]

    def __str__(self):
        return str(self.context)

    def __repr__(self):
        return str(self.context)


@dataclass(frozen=True)
class Agent:
    name: str
    prompt: Prompt

    def _prepare_prompt(self, context: ExecutionContext) -> Prompt:
        params = {v: context.get(v) for v in self.prompt.parameters}
        return self.prompt.apply(params)


class Graph:

    @classmethod
    def from_nodes(cls, nodes: set[Agent], edges: set[str]):
        graph = cls()
        for node in nodes:
            graph.add_node(node)

        node_id_to_node = {node.name: node for node in nodes}

        for edge in edges:
            from_node, to_node = edge.split("->")
            from_node = from_node.strip()
            to_node = to_node.strip()

            graph.add_edge(node_id_to_node[from_node], node_id_to_node[to_node])
        return graph

    def __init__(self):
        self.nodes = set()
        self.edges = dict()

    def add_node(self, value: Agent):
        self.nodes.add(value)
        self.edges[value] = []

    def add_edge(self, from_node: Agent, to_node: Agent):
        self.edges[from_node].append(to_node)

    def run(self, context: ExecutionContext):
        # sort the graph topologically
        visited: set[Agent] = set()
        stack: list[Agent] = []

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in self.edges[node]:
                dfs(neighbor)
            stack.append(node)

        for node in self.nodes:
            dfs(node)
        stack = stack[::-1]
        # run the graph

        bar = st.sidebar.progress(0)
        for i, node in enumerate(stack):
            context = node.run(context)
            bar.progress((i + 1) / len(stack))
        return context
