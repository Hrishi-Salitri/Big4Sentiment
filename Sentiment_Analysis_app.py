import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist


from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc

# ---------------------------start----------------------------------------------#

# Sheet for Deloitte
google_sheets_url = "https://docs.google.com/spreadsheets/d/1sQU07AjN8CnsdqhaUfvKWn2fpcb685xpmDDBv9H1s8g"
download_xl_suffix = "export?format=xlsx"
Deloitte_url = google_sheets_url + "/" + download_xl_suffix

# Sheet for KPMG
google_sheets_url = "https://docs.google.com/spreadsheets/d/1el99LapKcaqtl89oNCNuIec8tDWTY4_BT_4jswQ_RkA"
download_xl_suffix = "export?format=xlsx"
KPMG_url = google_sheets_url + "/" + download_xl_suffix

# Sheet for EY
google_sheets_url = "https://docs.google.com/spreadsheets/d/1THwvHOA2JBIrAoMBYIo8pSkbmLkFQO8CbH63FYT9yuQ"
download_xl_suffix = "export?format=xlsx"
EY_url = google_sheets_url + "/" + download_xl_suffix

# Sheet for PwC
google_sheets_url = "https://docs.google.com/spreadsheets/d/1M_1qDyuTba4KzRuw_SCY6M-aXKhwe7q18glvHvbX85o"
download_xl_suffix = "export?format=xlsx"
PwC_url = google_sheets_url + "/" + download_xl_suffix


Deloitte = pd.read_excel(Deloitte_url)
Kpmg = pd.read_excel(KPMG_url)
Ey = pd.read_excel(EY_url)
PwC = pd.read_excel(PwC_url)


app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

app.layout = html.Div(
    [
        html.Div(id="hidden-div", style={"display": "none"}),
        html.H1("Big 4 Sentiment Analysis", style={"textAlign": "center"}),
        html.H4(
            [
                "Welcome to the Big 4 consulting firms: ",
                dcc.RadioItems(
                    options=[
                        {"label": "Deloitte", "value": "deloitte"},
                        {"label": "KPMG", "value": "kpmg"},
                        {"label": "EY", "value": "ey"},
                        {"label": "PwC", "value": "pwc"},
                    ],
                    id="firm_radio",
                    inline=True,
                    value="deloitte",
                ),
            ],
            style={"textAlign": "center"},
        ),
        html.Br(),
        html.Div(
            children=[
                html.H5(
                    """Data Description: Our data will be looking at employee (located around the UK) reviews of the big four consulting companies.
            Data has been modeled to analyze sentiment, or feeling, of the company. Hope you enjoy!""",
                    style={"color": "white"},
                ),
            ],
            style={
                "backgroundColor": "#333",
                "padding": "20px",
            },
        ),
        html.Br(),
        html.Br(),
        dbc.Row(
            dcc.Dropdown(
                options=[
                    {"label": "Overall Rating", "value": "overall_rating"},
                    {
                        "label": "Work Life Balance",
                        "value": "work_life_balance",
                    },
                    {"label": "Culture", "value": "culture_values"},
                    {"label": "Diversity", "value": "diversity_inclusion"},
                    {"label": "Career Opportunity", "value": "career_opp"},
                    {
                        "label": "Compensation Benefits",
                        "value": "comp_benefits",
                    },
                    {"label": "Management", "value": "senior_mgmt"},
                ],
                value="overall_rating",
                style={
                    "backgroundColor": "#808080",
                    "color": "black",
                    "textAlign": "center",
                },
                id="rating-dropdown",
            ),
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H5(
                            """To the right we have a bar graph showing how employees rated their company. 
                                Toggle the dropdown menu to see different criteria the employees used to rate the company.
                                Scores are 1-5."""
                        ),
                        html.Br(),
                        html.Hr(),
                        html.Br(),
                        html.H5(
                            """Below we have a scatter plot describing the comments made about the firm. Using transfer learing from
                                the RoBERTa model, we took each comment and how positive or negative it was. This was done in perentages, 
                                so, for example, if a certain comment could be rated as 90% positive. Our data is split comments up into Pros
                                and Cons of the firm. Use the tabs to switch between the two types of comments. The scatter plot is filtered by 
                                the dropdown menu, and scores of those criteria. An example can be looking at comments from employees who 
                                rated work life balance between 1-2. Hover over any point to see their Positivity/Negativity score and 
                                what the employee had to say."""
                        ),
                    ]
                ),
                dbc.Col(dcc.Graph("rating-graph")),
            ]
        ),
        html.Br(),
        dbc.Row(
            [
                html.H5("Rating Score Slider", style={"textAlign": "center"}),
                html.Div(
                    [
                        dcc.RangeSlider(1, 5, 1, value=[1, 5], id="tab-slider"),
                    ]
                ),
            ],
        ),
        dbc.Row(
            [
                dcc.Tabs(
                    id="tabs",
                    value="pros-tab",
                    children=[
                        dcc.Tab(label="Pros", value="pros-tab"),
                        dcc.Tab(label="Cons", value="cons-tab"),
                    ],
                ),
                html.Div(id="tabs-content"),
            ],
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.H5("Best words employees used to describe the firm are..."),
                    style={"textAlign": "center"},
                ),
                dbc.Col(
                    html.H5("Worst words employees used to describe the firm are..."),
                    style={"textAlign": "center"},
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph("pro-wordcloud"), width=5),
                dbc.Col(
                    html.Div(
                        style={"border-left": "2px solid white", "height": "100%"}
                    ),
                    width=1,
                    style={
                        "display": "flex",
                        "align-items": "center",
                        "justify-content": "center",
                    },
                ),
                dbc.Col(dcc.Graph("con-wordcloud"), width=5),
            ]
        ),
        html.Br(),
    ]
)


# Callback to initialize the selected DataFrame based on the initial radio button value
@app.callback(
    Output("hidden-div", "children"),
    Input("firm_radio", "value"),
    # prevent_initial_call=True,  # Prevents the callback from running during app initialization
)
def initialize_dataframe(selected_firm):
    global df
    if selected_firm == "kpmg":
        df = Kpmg
    elif selected_firm == "ey":
        df = Ey
    elif selected_firm == "pwc":
        df = PwC
    else:
        df = Deloitte

    return "Initialized"


@app.callback(
    Output("rating-graph", "figure"),
    Input("rating-dropdown", "value"),
    Input("hidden-div", "children"),
)
def update_rating(rating, _):
    rating_counts = df[rating].value_counts().sort_index()
    fig = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        labels={rating: "Count"},
        template="plotly_dark",
    )
    fig.update_layout(xaxis_title="Score 1-5", yaxis_title="count")
    return fig


@app.callback(
    Output("pro-wordcloud", "figure"),
    Input("hidden-div", "children"),
)
def update_pro_wordcloud(_):
    text = " ".join(df["pros"])
    words = nltk.word_tokenize(text.lower())

    stop_words = set(stopwords.words("english"))
    filtered_words = [
        word for word in words if word.isalnum() and word not in stop_words
    ]

    fdist = FreqDist(filtered_words)
    top10words = fdist.most_common(10)
    top10words = dict(top10words)
    wordcloud = WordCloud(
        width=600, height=300, background_color="black", colormap="cool"
    ).generate_from_frequencies(top10words)

    fig = px.imshow(wordcloud.to_array(), binary_string=True, template="plotly_dark")
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    return fig


@app.callback(
    Output("con-wordcloud", "figure"),
    Input("hidden-div", "children"),
)
def update_con_wordcloud(_):
    text = " ".join(df["cons"])
    words = nltk.word_tokenize(text.lower())

    stop_words = set(stopwords.words("english"))
    filtered_words = [
        word for word in words if word.isalnum() and word not in stop_words
    ]

    fdist = FreqDist(filtered_words)
    top10words = fdist.most_common(10)
    top10words = dict(top10words)
    wordcloud = WordCloud(
        width=600, height=300, background_color="black", colormap="cool"
    ).generate_from_frequencies(top10words)

    # Display the word cloud using matplotlib
    fig = px.imshow(wordcloud.to_array(), binary_string=True, template="plotly_dark")
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    return fig


@app.callback(
    Output("tabs-content", "children"),
    Input("tabs", "value"),
    Input("hidden-div", "children"),
    Input("tab-slider", "value"),
    Input("rating-dropdown", "value"),
)
def Update_tab_scatter(tab, _, slider, dropdown):
    if tab == "pros-tab":
        df["pros_positive%"] = df["pros_positive%"].round(3)
        df["pros_negative%"] = df["pros_negative%"].round(3)

        min_value, max_value = slider

        filter_df = df[(df[dropdown] >= min_value) & (df[dropdown] <= max_value)]

        fig = px.scatter(
            filter_df,
            x="pros_positive%",
            y="pros_negative%",
            hover_data=["pros"],
            template="plotly_dark",
        )
        fig.update_layout(
            title="Percent of how Positive/Negative Pros(of the firm) comments were",
            yaxis_title="% Negative",
            xaxis_title="% Positive",
        )

        return dcc.Graph(figure=fig)
    elif tab == "cons-tab":
        df["cons_positive%"] = df["cons_positive%"].round(3)
        df["cons_negative%"] = df["cons_negative%"].round(3)

        min_value, max_value = slider

        filter_df = df[(df[dropdown] >= min_value) & (df[dropdown] <= max_value)]

        fig = px.scatter(
            filter_df,
            x="cons_positive%",
            y="cons_negative%",
            hover_data=["cons"],
            template="plotly_dark",
        )
        fig.update_layout(
            title="Percent of how Positive/Negative Cons(of the firm) comments were",
            yaxis_title="% Negative",
            xaxis_title="% Positive",
        )

        return dcc.Graph(figure=fig)


if __name__ == "__main__":
    app.run(debug=True)


# AttributeError: 'WordListCorpusReader' object has no attribute '_LazyCorpusLoader__args'
