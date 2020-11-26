

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import numpy as np
from pathlib import Path
from collections import defaultdict
from collections import OrderedDict
from rsarl.utils import get_mean_std
from rsarl.visualizer import gen_network_topology, gen_slot_table, gen_blocking_prob_line_graph

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
db = None

def set_callback(exp_name):
    @app.callback(
        [
            Output(f"{exp_name}:network-figure", 'figure'),
            Output(f"{exp_name}:slot-table", 'figure'),
            Output(f"{exp_name}:request-slider-label", 'children'),
        ],
        [
            Input(f"{exp_name}:request-slider", 'value'),
        ],
        [
            State(f"{exp_name}:network-figure", 'figure'),
            State(f"{exp_name}:slot-table", 'figure'),
            State(f"{exp_name}:request-slider-label", 'children'),
        ]
    )
    def update_act_history_figures(req_id, net_fig, slot_fig, slider_label):
        ctx = dash.callback_context
        # did not manipulate slider
        if not ctx.triggered or ctx.triggered[0]['value'] is None:
            return net_fig, slot_fig, slider_label

        slider_val_text = ctx.triggered[0]['prop_id'].split('.')[0]
        # getch experiment name
        exp_name = slider_val_text.split(":")[0]

        act, req, G = db.get_act_history(exp_name, req_id)

        # build figure
        net_figure = gen_network_topology(G, act)
        slot_figure = gen_slot_table(G, act)

        # build label
        request_slider_label = f"[{req_id}-th Request] source: {req.source} -> destination: {req.destination}, bandwidth: {req.bandwidth}"

        return net_figure, slot_figure, request_slider_label



@app.callback(
    [
        Output('exp-metric-tabs', 'active_tab'),
        Output('exp-metric-tabs', 'children'),
        Output('bp-line-graph', 'figure'),
        Output('agent-section', 'children'),
    ],
    [
        Input('experiment-select', 'value'),
    ],
    [
        State("agent-section", 'children'),
    ]
)
def insert_exp_info(exp_names: list, div):
    global db

    already_exist_exp_names = []
    already_exist_exp_req_id = []
    if div:
        # print(div[0].keys()) # dict_keys(['props', 'type', 'namespace'])
        # print(div[0]["props"]["children"][1]["props"]["children"][2]["props"]["children"][1]["props"]) 
        # # {'id': 'entropy:request-slider', 'marks': {'0': '0', '10000': '10000', '5000.0': '5000.0'}, 'value': 1329, 'min': 0, 'max': 10000, 'step': 1}
        if 'value' in div[0]["props"]["children"][1]["props"]["children"][2]["props"]["children"][1]["props"]:
            for d in div:
                exp_name = d["props"]["children"][0]["props"]["children"]
                req_id = d["props"]["children"][1]["props"]["children"][2]["props"]["children"][1]["props"]['value']
                already_exist_exp_names.append(exp_name)
                already_exist_exp_req_id.append(req_id)

    # build metric rows
    exp_metrics = []
    exp_sections = []
    first_exp_tab_id = None
    bp_figure = { "data": [], "layout": {"height": 200, "margin": dict(b=20,l=10,r=5,t=10),}, "frames": [], }

    # if list is not empty, insert target experimental info
    if exp_names:
        bp_list = []
        first_exp_tab_id = f'tabs-{exp_names[0]}'

        for i, exp_name in enumerate(exp_names):

            # ---- Agent Sections in Summary Section ---- #
            # if already exist, then display the same figure
            if exp_name in already_exist_exp_names:
                idx = already_exist_exp_names.index(exp_name)
                exp_sections.append(build_experiment(exp_name, req_id=already_exist_exp_req_id[idx]))
            else:
                exp_sections.append(build_experiment(exp_name))

            # ---- Experimental Settings in Summary Section ---- #
            env, net, agent, requester, hparam_dict = db.get_experiment_settings(exp_name)
            exp_dict = OrderedDict()
            exp_dict["Exp Name"] = exp_name
            exp_dict["Environment"] = env
            exp_dict["Network"] = net
            exp_dict["Agent Name"] = agent
            exp_dict["Request Type"] = requester
            exp_dict.update(hparam_dict)
            # generate table
            exp_metrics.append(build_exp_setting_table(exp_dict))

            # ---- Blocking Probability in Summary Section ---- #
            # calc blocking prob
            batches = db.get_batches(exp_name)
            bp_per_batch = db.get_bp_per_batch(exp_name, batches)
            # prepare coordinates for list of line graph
            x = np.array(list(bp_per_batch.keys()))
            y_mean, y_std = get_mean_std(bp_per_batch)
            bp_list.append((exp_name, x, y_mean, y_std))

        # build blocking prob figure
        bp_figure = gen_blocking_prob_line_graph(bp_list)

    return first_exp_tab_id, exp_metrics, bp_figure, exp_sections


def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.H6("RSA-RL: Visualizer"),
        ],
    )

def build_graph_title(title):
    return html.P(className="graph-title", children=title)

def build_header(db):
    return html.Div(
        className="row",
        id="top-row-header",
        children=[
            dbc.Col(
                html.Div(
                    id="header-container",
                    children=[
                        build_banner(),
                        build_graph_title("Select Experiments"),
                        dcc.Dropdown(
                            id="experiment-select",
                            # options to be selected by users
                            # label appear inside selector
                            options=[
                                {"label": i, "value": i}
                                for i in db.get_experiment_names()
                            ],
                            multi=True,
                            # initial values inside select operator
                            # for our objective, here should be empty...
                            value=[],
                        ),
                    ],
                ), 
                width=12
            ),
        ],
    )

def build_slider(exp_name: str, label: str, req_id: int=0):
    slider_id = f"{exp_name}:request-slider"
    slider_label_id = f"{exp_name}:request-slider-label"
    range_min = 0
    range_max = 10000
    return html.Div(
            className="twelve columns slider",
            children=[
                html.P(id=slider_label_id, className="graph-title", children=label),
                # Adjust the Spacecraft Parameters
                dcc.Slider(
                    # id={'type': 'req-slider', 'index': slider_id},
                    id=slider_id,
                    min=range_min,
                    max=range_max,
                    value=req_id,
                    step=1,
                    marks={
                        range_min: f"{range_min}", 
                        range_max/2: f"{range_max/2}", 
                        range_max: f"{range_max}",
                    },
                )
            ],
        )  

def build_section_barner(title: str):
    return html.Div(
                className="section-banner",
                children=title,
            )

def build_table_barner(title: str):
    return html.Div(
                className="table-banner",
                children=title,
            )

def build_exp_setting_table(exp: dict):
    name = exp["Exp Name"].replace("_", "-")
    # table header
    table_header = [
        html.Thead(html.Tr([
            html.Th("ParamName", className="table-row"), 
            html.Th("Value", className="table-row")
        ]))
    ]
    # table rows
    table_rows = []
    for k, v in exp.items():
        table_rows.append(html.Tr([
            html.Td(k, className="table-row"), 
            html.Td(v, className="table-row")
        ]))

    table_body = [html.Tbody(table_rows)]
    table = dbc.Table(
        table_header + table_body,
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
    )

    content = html.Div(
        [table],
        className="table-content",
        )

    tab = dbc.Tab(
        tab_id=f'tabs-{exp["Exp Name"]}',
        label=name,
        children=[content]
    )
    
    return tab


def build_summary():
    return html.Div(
            className="pretty_container twelve columns",
            children=[
                # agent name
                build_section_barner("Summary"),
                html.Div(
                    className="figure_container twelve columns",
                    children=[
                        # experimental setting table
                        html.Div(
                            id="metric-table",
                            className="seven columns plot-container plotly",
                            children=[
                                build_table_barner("Experimental Settings"),
                                dbc.Tabs(
                                    [],
                                    id="exp-metric-tabs",
                                    active_tab="None",
                                ),
                                html.Div(id="tab-content"),
                            ]
                        ),
                        # linear figure
                        html.Div(
                            id="bp-comparison",
                            className="five columns plot-container plotly",
                            children=[
                                build_table_barner("Blocking Probability Comparison"),
                                dcc.Loading(
                                    children=dcc.Graph(id="bp-line-graph")
                                ),
                            ],
                            
                        ),
                    ],
                ),
            ],
        )


def build_experiment(exp_name: str, req_id: int=0):
    # generate initial figure (first seed, request-id is 0)
    act, req, G = db.get_act_history(exp_name, req_id)
    # build figure
    net_figure = gen_network_topology(G, act)
    slot_figure = gen_slot_table(G, act)
    # build label
    request_slider_label = f"[{req_id}-th Request] source: {req.source} -> destination: {req.destination}, bandwidth: {req.bandwidth}"

    return html.Div(
            id=f"{exp_name}:section",
            className="pretty_container twelve columns",
            children=[
                # agent name
                build_section_barner(exp_name),
                html.Div(
                    className="figure_container twelve columns",
                    children=[
                        # network topology
                        html.Div(
                            className="three columns plot-container plotly",
                            children=dcc.Loading(
                                children=dcc.Graph(
                                    figure=net_figure,
                                    id=f"{exp_name}:network-figure",
                                ),
                            ),
                        ),
                        # slot table
                        html.Div(
                            className="nine columns plot-container plotly",
                            children=dcc.Loading(
                                children=dcc.Graph(
                                    figure=slot_figure,
                                    id=f"{exp_name}:slot-table", 
                                ),
                            ),
                        ),
                        # slider
                        build_slider(exp_name, request_slider_label, req_id),
                    ],
                ),
            ],
        )

def build_dash(local_db):
    """
    Make HTML
    """
    global db
    db = local_db
    all_exp_names = db.get_experiment_names()
    # html layout
    app.layout = html.Div(children=[
        # start one row
        html.Div(
            id="body",
            className="body twelve columns",
            children=[
                # header section
                build_header(db),
                # summary section
                html.Div(
                    id = "summary-section",
                    className="container scalable",
                    children=[
                        build_summary(),
                    ],
                ),
                # agent section
                html.Div(
                    id = "agent-section",
                    className="container scalable",
                    children=[
                        # each agent section
                        build_experiment(exp_name) for exp_name in all_exp_names
                    ],
                ),
            ],
        ),
        # tmp divs
        html.Div(id='check_current_dom_order_callback', style={'display': 'none'}),
    ])

    # register callbacks
    for exp_name in all_exp_names:
        set_callback(exp_name)

    return app

