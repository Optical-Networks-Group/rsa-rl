
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from rsarl.utils import path_to_edges

# color map
cmap = plt.get_cmap("tab10")

def gen_slot_table(G, act):
    edge_dict = nx.get_edge_attributes(G, name="slot")
    n_slot = len(list(edge_dict.values())[0])
    
    # assigned path
    if act.path is not None:
        assigned_edges = path_to_edges(act.path)

    # edge-name list
    edges = []
    slot_data=[]
    duration_data=[]
    # slot table values (1: available 0: occupied)
    for e, attr in G.edges.items():
        slot = attr["slot"]
        if act.path is not None and e in assigned_edges:
            for s in range(act.slot_idx, act.slot_idx + act.n_slot):
                slot[s] = 0.5
        
        edges.append(f"{e}")
        slot_data.append(slot)
        duration_data.append(attr["time"])

    # make heatmap
    # slot utilization table
    slot_util = go.Heatmap(
        name="slot-util",
        x=list(range(n_slot)),
        y=edges,
        z=slot_data,
        colorscale="BuGn",
        reversescale=True,
        opacity=1,
        xgap=2,
        ygap=5,
        showscale=False
    )
    # path duration time table
    duration = go.Heatmap(
        name="duration",
        x=list(range(n_slot)),
        y=edges,
        z=duration_data,
        opacity=0.3,
        colorscale="Greys",
        xgap=2,
        ygap=5,
        showscale=False
    )

    # adjust layout
    layout = go.Layout(
        margin=dict(l=20, r=10, t=10, b=0),
        xaxis=dict(
            showgrid=False
        ),
        yaxis=dict(
            showgrid=False
        )
    )
    return go.Figure(data=[slot_util, duration], layout=layout)


def gen_network_topology(G, act):
    node_pos_dict = nx.get_node_attributes(G, name='position')
    """
        Add edges
    """
    # assigned path
    if act.path is not None:
        path_edges = path_to_edges(act.path)

    # Add edges
    edge_x = []
    edge_y = []
    for e in G.edges():
        if act.path is not None and e in path_edges:
            continue

        x0, y0 = node_pos_dict[e[0]]
        x1, y1 = node_pos_dict[e[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Add colored edges
    if act.path is not None:
        color_edge_x = []
        color_edge_y = []
        for e in path_edges:
            x0, y0 = node_pos_dict[e[0]]
            x1, y1 = node_pos_dict[e[1]]
            color_edge_x.append(x0)
            color_edge_x.append(x1)
            color_edge_x.append(None)
            color_edge_y.append(y0)
            color_edge_y.append(y1)
            color_edge_y.append(None)
        
        colored_edge_trace = go.Scatter(
            x=color_edge_x, y=color_edge_y,
            line=dict(width=3.0, color='#f1b512'),
            hoverinfo='none',
            mode='lines')

        edge_trace = [colored_edge_trace] + [edge_trace]
    else:
        edge_trace = [edge_trace]

    """
        Add nodes
    """
    node_x = []
    node_y = []
    for _, pos in node_pos_dict.items():
        x, y = pos
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        showlegend=False,
        marker=dict(
            showscale=False, 
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            line_width=2)
        )
    """
        Hover nodes
    """
    node_text = []
    for n in G.nodes():
        node_text.append(f'Node id: {n}')
    node_trace.text = node_text

    """
        Make layout
    """
    layout=go.Layout(
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=20),
        annotations=[dict(
            text="",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002 )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    
    return go.Figure(data=edge_trace + [node_trace], layout=layout)


def gen_blocking_prob_line_graph(bp_list: list):
    data = []
    # search longest x
    base_exp_name = ""
    shared_x = []
    for exp_name, x, _, _ in bp_list:
        if len(shared_x) < len(x):
            base_exp_name = exp_name
            shared_x = x

    # padding empty coodinate of x-axis
    for i in range(len(bp_list)):
        exp_name, _, y_mean, y_std = bp_list[i]
        if exp_name != base_exp_name:
            if len(y_std) == 1: # heuristic methods
                y_std  = np.ones_like(shared_x) * y_std
                y_mean = np.ones_like(shared_x) * y_mean
                bp_list[i] = exp_name, shared_x, y_mean, y_std
            
            else: # RL methods
                _y_std  = np.ones_like(shared_x) * y_std[-1]
                _y_std[:len(y_std)] = y_std

                _y_mean = np.ones_like(shared_x) * y_mean[-1]
                _y_mean[:len(y_mean)] = y_mean

                bp_list[i] = exp_name, shared_x, _y_mean, _y_std
    
    # build figures
    for i, (exp_name, _, y_mean, y_std) in enumerate(bp_list):
        # decide color
        r, g, b = cmap(i)[:3]
        color = (int(r*255), int(g*255), int(b*255))
        alpha_color = (int(r*255), int(g*255), int(b*255), 0.3)

        trace_upper = go.Scatter(
            name = f"upper-{exp_name}",
            x = shared_x, 
            y = y_mean + y_std,
            line=dict(width=0),
            fillcolor = f"rgba{alpha_color}",
            fill = "tonexty",
            showlegend=False,
        )
        trace_mean = go.Scatter(
            name = exp_name,
            x = shared_x, 
            y = y_mean,
            line=dict(color=f"rgb{color}"),
            fillcolor = f"rgba{alpha_color}",
            fill = "tonexty",
            showlegend=True
        )
        trace_lower = go.Scatter(
            name = f"lower-{exp_name}",
            x = shared_x, 
            y = y_mean - y_std,
            line=dict(width=0),
            showlegend=False,
        )
        # data: NOTE that don't change this order
        data.append(trace_lower)
        data.append(trace_mean)
        data.append(trace_upper)
    
    # layout
    layout=go.Layout(
        titlefont_size=16,
        # TODO: change heights depending on the number of EXPs. 
        height=400,
        hovermode='closest',
        margin=dict(b=20,l=10,r=5,t=10),
        xaxis_title='Batch [epoch]',
        yaxis_title='Blocking Probability [%]',
        legend_title_text='EXP. Name',
        legend=dict(
            xanchor="right",
            x=0.99,
            y=0.99,
            traceorder="reversed",
            title_font_family="Open Sans",
            font=dict(
                family="Open Sans",
                size=14,
                color="black"
            ),
            bgcolor="White",
            bordercolor="Black",
            borderwidth=2
        ),
    )
    return go.Figure(data=data, layout=layout)

