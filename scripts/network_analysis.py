import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re
import os

def build_user_mention_network(df, min_mentions=5):
    print("\nMention ağı oluşturuluyor...")

    G = nx.DiGraph()

    for _, row in df.iterrows():
        source_user = row["user_name"]
        text = row["text"]

        mentions = re.findall(r"@(\w+)", text)  

        for mentioned_user in mentions:
            G.add_edge(source_user, mentioned_user)

    node_degrees = dict(G.degree())
    important_nodes = [node for node, degree in node_degrees.items() if degree >= min_mentions]
    G = G.subgraph(important_nodes).copy()

    if not os.path.exists("visualizations/network_graphs"):
        os.makedirs("visualizations/network_graphs")

    plt.figure(figsize=(18, 18))
    pos = nx.spring_layout(G, k=0.15)
    nx.draw(G, pos,
            with_labels=True,
            node_size=50,
            node_color="skyblue",
            edge_color="gray",
            font_size=6,
            arrowsize=10)
    plt.title("Kullanıcılar Arası Mention Ağı", fontsize=20)
    plt.savefig("visualizations/network_graphs/user_mention_network.png")
    plt.show()

    print("Mention ağı çizildi ve kaydedildi: visualizations/network_graphs/user_mention_network.png")
