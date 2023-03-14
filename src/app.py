import pandas as pd

# dash
from jupyter_dash import JupyterDash
from dash import html, dcc
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import no_update


pd.set_option('display.max_columns', None)


import os

csvs_byte = []
directory_in_str = 'D:\\Jupyter\\Untitled Folder\\trade_prices'
directory = os.fsencode(directory_in_str)

for file in os.listdir(directory):
    csvs_byte.append(file)

# file setup, copied from kaggle notebook
csvs = []
for csv in csvs_byte:
    csvs.append(csv.decode('utf-8'))

jp = [pd.read_csv(file_name, on_bad_lines='skip') for file_name in csvs]  # list of dataframes, for workability sake

jp_recent = []
for df in jp:
    df = df.loc[df.Year >= 2016].reset_index(drop=True)
    jp_recent.append(df)

pref_code_df = pd.read_csv('prefecture_code.csv')

pref_list = pref_code_df.EnName.unique().tolist()

pref_codes_top5 = [1, 2, 4, 40, 44]
jp5 = [jp_recent[code] for code in pref_codes_top5]
jp5 = pd.concat(jp5)

jp5_all = [jp[code] for code in pref_codes_top5]
jp5_all = pd.concat(jp5_all)


# converting columns
convert_columns = ['TradePrice', 'Area', 'UnitPrice']

for df in jp_recent:
    for col in convert_columns:
        df[col] = pd.to_numeric(df[col])

# Mean Prices
mean_price_list = []
mean_size_list = []
mean_trade_price_list = []
for dataframe in jp_recent:
    mean_price_list.append(dataframe['UnitPrice'].mean())
    mean_size_list.append(dataframe['Area'].mean())
    mean_trade_price_list.append(dataframe['TradePrice'].mean())
mean_price_list = [mean_price * 0.0076 for mean_price in mean_price_list]  # converting from Yen to USD
mean_trade_price_list = [mean_trade_price * 0.0076 for mean_trade_price in mean_trade_price_list]
mean_prices = pd.DataFrame()

# setting up df columns
mean_prices['Prefecture'] = np.array(pref_list)
mean_prices['Trade Price'] = np.array(mean_trade_price_list)
mean_prices['Unit Price'] = np.array(mean_price_list)
mean_prices['Lot Size'] = np.array(mean_size_list)
mean_prices['Lot Size / Unit Price'] = mean_prices['Lot Size'] / mean_prices['Unit Price']
mean_prices.sort_values(by='Lot Size / Unit Price', ascending=False).round(2)

mean_prices5 = mean_prices.sort_values(by='Lot Size / Unit Price', ascending=False).round(2).iloc[0:5]

jp_recent = pd.concat(jp_recent)
jp_recent.groupby('Prefecture')[['TradePrice', 'Area']].agg('mean').reset_index()

# Price Metrics
# trade prices
tprice_tab = pd.DataFrame(jp_recent.groupby('Prefecture')[['TradePrice', 'Area']].agg('mean').reset_index().sort_values(by='TradePrice'))
tprice_tab['TradePrice'] = tprice_tab['TradePrice'] * 0.0076  # yen to usd
tprice_tab[['TradePrice', 'Area']] = tprice_tab[['TradePrice', 'Area']].round(0)
tprice_tab.sort_values(by='TradePrice', ascending=False, inplace=True)

# unit prices
uprice_tab = pd.DataFrame(jp_recent.groupby('Prefecture')[['UnitPrice', 'Area']].agg('mean').reset_index().sort_values(by='UnitPrice'))
uprice_tab['UnitPrice'] = uprice_tab['UnitPrice'] * 0.0076  # yen to usd
uprice_tab[['UnitPrice', 'Area']] = uprice_tab[['UnitPrice', 'Area']].round(0)
uprice_tab.sort_values(by='UnitPrice', ascending=False, inplace=True)
uprice_tab

# lot size / unit price
luprice_tab = pd.DataFrame(jp_recent.groupby('Prefecture')[['UnitPrice', 'Area']].agg('mean').reset_index().sort_values(by='UnitPrice'))
luprice_tab['UnitPrice'] = luprice_tab['UnitPrice'] * 0.0076
luprice_tab['Area / UnitPrice'] = luprice_tab['Area'] / luprice_tab['UnitPrice']
luprice_tab['Area / UnitPrice'] = luprice_tab['Area / UnitPrice'].round(2)
luprice_tab.sort_values(by='Area / UnitPrice', ascending=True, inplace=True)
luprice_tab


# surrounding characteristics of each prefecture
char = jp5_all.groupby(['Prefecture', 'Region']).size().to_frame('Count').reset_index()
type = jp5_all.groupby(['Prefecture', 'Type']).size().to_frame('Count').reset_index()


# avg home prices for each municipality, within each prefecture
jp5['TradePrice'] = jp5['TradePrice']*0.0076  # Yen to USD exchange rate
prices = pd.DataFrame(jp5.groupby(['Prefecture', 'Municipality'])['TradePrice'].agg('mean'))
prices['TradePrice'] = prices['TradePrice'].round(0)
prices.reset_index(inplace=True)
prices

# Metrics for Each Prefecture
metrics = pd.DataFrame()
metrics['Build Year'] = jp5.groupby(['Prefecture'])['BuildingYear'].agg('mean').round(0)
metrics.reset_index(inplace=True)
metrics['Central Station'] = jp5.groupby('Prefecture')['NearestStation'].agg(pd.Series.mode).reset_index()['NearestStation']
metrics['Area (m^2)'] = (jp5.groupby('Prefecture')['Area'].agg('mean')).round(0).reset_index()['Area']

# dash
app = Dash(external_stylesheets=[dbc.themes.JOURNAL])
server = app.server

colors = {
    'background': '#faedcd',
    'text': '#1b263b',
    'header': '#353535',
    'box': '#f5ebe0'
}

# sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# to account for sidebar
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem"
}

sidebar = html.Div(
    [
        html.H2("Cheap Houses: Japan"),
        html.Hr(),
        html.P(
            'Inspired by @cheaphousesjapan, this project is designed to take a closer look at the Japanese housing market from a foreign lens. Many Japanese houses are listed for much cheaper than the typical US house, especially in non-tourist locations. Through this project, we can analyze which prefectures are the best starting points for a foreign homebuyer with an eye for cheap houses.'),
    ],
    style=SIDEBAR_STYLE,

)

prefecture_carousel = dbc.Carousel(items=[
    {'key': '1', 'src': 'https://tinyurl.com/y2y73c42', 'header': 'from cheaphousesjapan'},
    {'key': '2', 'src': 'https://tinyurl.com/ycxwda85'},
    {'key': '3', 'src': 'https://tinyurl.com/yeym5628'},
    {'key': '4', 'src': 'https://tinyurl.com/2dabka52'},
    {'key': '5', 'src': 'https://tinyurl.com/3r28cmte'},
    {'key': '6', 'src': 'https://tinyurl.com/3ey2n5sn'},
    {'key': '7', 'src': 'https://tinyurl.com/3ebuxt66'}
], class_name='w-50', interval='3000', indicators=False)

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[

    sidebar,

    # intro
    html.Div([
        html.H1("Cheap Houses in Japan", style={
            "margin-left": "18rem",
            "margin-right": "2rem",
            "padding": "2rem 1rem"
        }),

        dbc.Card([
            html.P(
                "In the US, a cheap house is a small house - but in Japan, cheap houses are sometimes big houses. Contrary to many parts of the US, houses in Japan tend to depreciate over time during their expected lifespan of 15-20 years, which is partially due to a housing culture that pushes people to buy new. Houses are particularly cheap in rural areas, where it's not uncommon to find property listings for less than $50,000 - or $500. As you might expect, some of these properties require significant renovations to the building's structure, appliances, or plumbing, but some homebuyers might still see potential in these properties as either vacation or retirement homes. And depending on the prefecture, renovations can even be funded through government grants. The question then becomes: Where should I start? "
                , style=CONTENT_STYLE, className='lead')
        ], class_name='border border-3 border-warning',
            style={
                "margin-left": "18rem",
                "margin-right": "2rem",
                "padding": "2rem 1rem",
                'background': colors['background']}
        )
    ]),

    # trade price, unit price, custom metric graph and tab
    html.Div([
        html.H2('Cost vs Size', style=CONTENT_STYLE)
    ]),

    html.Div([
        dcc.Tabs(id='price tabs', value='trade price graph', children=[
            dcc.Tab(label='Trade Price', value='trade price graph'),
            dcc.Tab(label='Unit Price', value='unit price graph'),
            dcc.Tab(label='Lot Size / Unit Price', value='size price graph')
        ]),
    ], style=CONTENT_STYLE),

    html.Div([
        html.H4(id='tab_fig_title')
    ], style={
        "margin-left": "18rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem"
    }),

    html.Div(children=[
        dbc.Row([
            dbc.Col(
                html.Div([
                    dcc.Graph(id='tab_fig', style={'width': '65vh', 'height': '65vh'}),
                ], style={'display': 'inline-block'})
            ),

            dbc.Col(
                dbc.Card(
                    [
                        dbc.Row([
                            html.Div([
                                html.P(id='tab_comment', className='lead')
                            ])
                        ]),

                        dbc.Row([
                            html.Div([
                                prefecture_carousel
                            ])
                        ], style={'margin-left': '175px', 'margin-top': '50px', 'width': '100%'})
                    ],
                    class_name='border rounded-3 p-5 border-warning border-3',
                    style={'background': colors['background']}
                )
            )

        ]), ], style={"margin-left": "18rem",
                      "margin-right": "2rem",
                      "padding": "2rem 1rem",
                      'display': 'flex'}),

    html.H2('Prefecture Statistics', style={
        "margin-left": "18rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
        'color': colors['text'],
    }),

    # dropdown
    html.Div(children=[
        html.Label('Choose a Prefecture:'),
        dcc.Dropdown(
            options=[
                {'label': 'Aomori', 'value': 'Aomori Prefecture'},
                {'label': 'Akita', 'value': 'Akita Prefecture'},
                {'label': 'Saga', 'value': 'Saga Prefecture'},
                {'label': 'Miyazaki', 'value': 'Miyazaki Prefecture'},
                {'label': 'Iwate', 'value': 'Iwate Prefecture'}
            ],
            id='dropdown', value='Aomori Prefecture')
    ], style={
        "margin-left": "18rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
        'color': colors['text'],
        'backgroundColor': colors['background']
    }),

    html.Div([

        dbc.Row([

            # cards and comments
            dbc.Col(

                [
                    html.Div([
                        dbc.CardGroup([
                            dbc.Card(
                                dbc.CardBody([
                                    html.H5('Average Home Size (m^2)', className='card-title'),
                                    html.P(id='lot size', className='card-text')
                                ])),
                            dbc.Card(
                                dbc.CardBody([
                                    html.H5('Average Build Year', className='card-title'),
                                    html.P(id='build year', className='card-text')
                                ])),
                            dbc.Card(
                                dbc.CardBody([
                                    html.H5('Central Station', className='card-title'),
                                    html.P(id='central station', className='card-text')
                                ]))
                        ])
                    ]),

                    # comments paragraph
                    html.Div([
                        dbc.Card([
                            html.P(
                                "We can zoom our search in further by dividing up each prefecture into their municipalities. We can see that each prefecture has a few municipalities with ridiculously low home prices, most of which will undoubtedly require significant renovations. Even for higher cost of living municipalities, average sale prices don't surpass 200k.",
                                className='lead'),
                            html.Br(),
                            html.P(
                                "Different prefectures also have different land characteristics. Region characteristics are similar across prefectures, with 70-80% residential areas and slightly differeing amounts of commerical and industrial areas. What's more interesting is the type of land that each home is on. Aomori, Saga, and Iwate are around 30-38% agricultural or forest land with 60-65% residential land, indicating that these prefectures might be suitable for city/suburb life. Meanwhile, Akita and Miyazaki are around 40-45% agricultural or forest land with 50-55% residential land, which would be better for those interested in the peaceful, country life.",
                                className='lead')
                        ], class_name='border border-3 border-warning p-5',
                            style={'background': colors['background'],
                                   'margin-top': '50px'}
                        )
                    ])
                ],
                style=CONTENT_STYLE
            ),

            # graphs
            dbc.Col(
                [
                    dbc.Card([
                        dbc.Col(
                            [
                                dbc.Row([
                                    dbc.Col(
                                        dcc.Graph(id='region')
                                    ),
                                    dbc.Col(
                                        dcc.Graph(id='land type'),
                                    )
                                ], class_name='g-0'),

                                dbc.Row([
                                    dcc.Graph(id='municipality prices')
                                ])
                            ]
                        )
                    ])
                ])
        ])
    ]),

])


@app.callback(
    [
        Output('tab_fig_title', 'children'),
        Output('tab_fig', 'figure'),
        Output('tab_comment', 'children')
    ],
    [
        Input('price tabs', 'value')
    ])
def update_tabs(tab):
    if tab == 'trade price graph':
        tab_fig_title = 'Average Trade Prices ($)'
        tab_comment = 'Looking at the average sale price of homes in each prefecture, we might think that Akita, Shimane, Aomori, Tottori, and Miyazaki are the best places to look. This might be true if we were only interested in price, but what about the size of the home? Looking at unit prices, or the dollar cost per square meter, we can get a sense of which prefectures are the most cost efficient.'
        tab_fig = px.bar(tprice_tab, y='Prefecture', x='TradePrice', color_discrete_sequence=['#52796f'])
        tab_fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'])
        tab_fig.update_yaxes(type='category', range=[42, 46])  # zoom on top 5 cheapest prefectures
        tab_fig.update_xaxes(range=[0, 100000])

    if tab == 'unit price graph':
        tab_fig_title = 'Unit Prices ($ / m^2)'
        tab_fig = px.bar(uprice_tab, y='Prefecture', x='UnitPrice', color_discrete_sequence=['#52796f'])
        tab_comment = 'The list for unit prices is similar, but we now see that Yamanashi, Ibaraki, and Saga are also good choices for the cost-efficient homebuyer. '
        tab_fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'])
        tab_fig.update_yaxes(type='category', range=[42, 46])
        tab_fig.update_xaxes(range=[0, 200])

    if tab == 'size price graph':
        tab_fig_title = 'Area / Unit Price'
        tab_fig = px.bar(luprice_tab, y='Prefecture', x='Area / UnitPrice', color_discrete_sequence=['#52796f'])
        tab_comment = 'By creating a new metric, we can find which prefectures have the best of both worlds; dividing the average lot size by the average unit price will tell us which prefectures have a combination of high average lot size and low average unit price. These prefectures will be the top five starting points for homebuyers interested in nothing but getting the largest house they can, while also spending as little as possible.'
        tab_fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'])
        tab_fig.update_yaxes(type='category', range=[42, 46])

    return tab_fig_title, tab_fig, tab_comment


@app.callback(
    [
        Output('municipality prices', 'figure'),
        Output('region', 'figure'),
        Output('land type', 'figure'),
        Output('build year', 'children'),
        Output('central station', 'children'),
        Output('lot size', 'children')
    ],
    [
        Input('dropdown', 'value')
    ]
)
def update_figure(prefecture):
    prices_filtered = prices[prices.Prefecture == prefecture]
    char_filtered = char[char.Prefecture == prefecture]
    type_filtered = type[type.Prefecture == prefecture]

    # creating figures
    fig_price = px.bar(prices_filtered, y='Municipality', x='TradePrice', color_discrete_sequence=['#52796f'],
                       title='Average Home Price per Municipality (USD)')

    fig_region = px.pie(char_filtered, values='Count', names='Region', title='Region Characteristics')

    fig_type = px.pie(type_filtered, values='Count', names='Type', title='Home Land Type')

    fig_price.update_layout(
        yaxis={'categoryorder': 'total descending'},
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )

    fig_price.update_yaxes(type='category',
                           range=[len(prices_filtered.Municipality) - 5, len(prices_filtered.Municipality) - 1])
    fig_price.update_xaxes(range=[0, 75000])

    # update region
    fig_region.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'])

    fig_region.update_traces(
        marker=dict(colors=['#f6bd60', '#f7ede2', '#f5cac3', '#84a59d', '#f28482'])
    )

    fig_region.update_layout(margin=dict(t=0, b=0, l=0, r=0))

    # update type
    fig_type.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'])

    fig_type.update_traces(
        marker=dict(colors=['#f6bd60', '#f7ede2', '#f5cac3', '#84a59d', '#a3b18a'])
    )

    fig_type.update_layout(margin=dict(t=0, b=0, l=0, r=0))

    # metrics
    metrics_filtered = metrics[metrics.Prefecture == prefecture]

    build_year_str = str(metrics_filtered['Build Year'].values[0])
    central_station_str = str(metrics_filtered['Central Station'].values[0])
    lot_size_str = str(metrics_filtered['Area (m^2)'].values[0])

    return fig_price, fig_region, fig_type, build_year_str, central_station_str, lot_size_str


if __name__ == '__main__':
    app.run_server()
